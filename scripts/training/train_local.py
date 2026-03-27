# This file is deprecated. You can run training natively through the unified train.py.
""""
Optimized for RTX 4060 Ti (8GB VRAM)

Usage:
    python train_local.py --data_path ./data/kaggle_resume_pdf

Requirements:
    pip install -r requirements.txt
    
For Windows with CUDA:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""

import os
import sys
import argparse
import time
import random
import json
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torchcrf import CRF
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score


# =====================
# Configuration
# =====================
ENTITY_LABELS = [
    "O", "B-SKILL", "I-SKILL", "B-EXP", "I-EXP", "B-EDU", "I-EDU",
    "B-PROJ", "I-PROJ", "B-ACH", "I-ACH", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-DATE", "I-DATE", "B-NAME", "I-NAME",
    "B-CONTACT", "I-CONTACT", "B-CERT", "I-CERT"
]

LABEL2ID = {label: idx for idx, label in enumerate(ENTITY_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(ENTITY_LABELS)


# =====================
# Model Architecture
# =====================
class BertBiLSTMCRF(nn.Module):
    """BERT + BiLSTM + CRF for Resume NER"""
    
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        # BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        bert_hidden_size = self.bert.config.hidden_size
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(lstm_dropout)
        self.hidden2tag = nn.Linear(lstm_hidden_size * 2, NUM_LABELS)
        self.crf = CRF(NUM_LABELS, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        emissions = self.hidden2tag(lstm_output)
        
        outputs = {"emissions": emissions}
        
        if labels is not None:
            mask = attention_mask.bool() & (labels != -100)
            labels_for_crf = labels.clone()
            labels_for_crf[labels == -100] = 0
            
            # Run CRF in float32 to prevent FP16 overflow and NaN gradients
            with autocast(device_type='cuda', enabled=False):
                loss = -self.crf(emissions.float(), labels_for_crf, mask=mask, reduction='mean')
            outputs["loss"] = loss
        
        with torch.no_grad():
            mask = (attention_mask.bool() & (labels != -100)) if labels is not None else attention_mask.bool()
            with autocast(device_type='cuda', enabled=False):
                predictions = self.crf.decode(emissions.float(), mask=mask)
            outputs["predictions"] = predictions
            
        return outputs


# =====================
# Dataset
# =====================
class ResumeNERDataset(Dataset):
    """PyTorch Dataset for Resume NER"""
    
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens, labels = self.examples[idx]
        
        encoding = self.tokenizer(
            tokens, is_split_into_words=True,
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        
        word_ids = encoding.word_ids()
        aligned_labels = []
        prev_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != prev_word_idx:
                if word_idx < len(labels):
                    aligned_labels.append(LABEL2ID.get(labels[word_idx], LABEL2ID["O"]))
                else:
                    aligned_labels.append(LABEL2ID["O"])
            else:
                if word_idx < len(labels):
                    label = labels[word_idx]
                    if label.startswith("B-"):
                        label = "I-" + label[2:]
                    aligned_labels.append(LABEL2ID.get(label, LABEL2ID["O"]))
                else:
                    aligned_labels.append(LABEL2ID["O"])
            prev_word_idx = word_idx
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }


# =====================
# Data Loading
# =====================
def load_resume_data(data_path: str):
    """Load resumes from directory (PDF, TXT, CSV)"""
    import re
    import glob
    
    examples = []
    
    def weak_label_text(text: str):
        """Create improved weak BIO labels from raw resume text."""
        if not text or len(text.strip()) < 20:
            return None

        tokens = re.findall(r"\b\w+[\w+.+-]*\b|[^\w\s]", text)

        if len(tokens) < 5:
            return None

        labels = ["O"] * len(tokens)

        # Expanded vocabularies
        skill_terms = {
            # IT & Tech
            "python", "javascript", "typescript", "java", "c++", "c#", "react", "react.js",
            "node", "nodejs", "express", "nextjs", "html", "css", "tailwind", "vue",
            "bootstrap", "pytorch", "tensorflow", "keras", "scikit-learn", "numpy",
            "pandas", "sql", "mongodb", "postgresql", "mysql", "docker", "aws",
            "azure", "gcp", "kubernetes", "nlp", "machine", "learning", "deep",
            "git", "linux", "rest", "graphql", "redis", "elasticsearch", "opencv",
            "firebase", "supabase", "swift", "kotlin", "flutter", "dart", "rust", "go",
            "hadoop", "spark", "hive", "devops", "ci/cd", "jenkins", "jira",
            # Finance, Accounting & Banking
            "accounting", "audit", "tax", "tally", "financial", "budget", "payroll",
            "bookkeeping", "reconciliation", "invoice", "banking", "kyc", "aml",
            # HR & Management
            "recruitment", "onboarding", "sourcing", "hr", "agile", "scrum",
            "management", "leadership", "operations", "strategy", "planning", "bpo",
            # Sales & Marketing
            "sales", "b2b", "b2c", "marketing", "seo", "sem", "lead", "generation",
            "crm", "salesforce", "advertising", "branding", "campaign",
            # Design, Art & Architecture
            "photoshop", "illustrator", "figma", "ui", "ux", "autocad", "revit",
            "design", "architecture", "interior", "3d", "rendering",
            # Engineering (Civil, Mech, Electrical) & Other
            "civil", "mechanical", "electrical", "hvac", "plc", "cad", "structural",
            "thermodynamics", "manufacturing", "quality", "control", "maintenance",
            # Legal, Healthcare & Agriculture
            "legal", "litigation", "drafting", "compliance", "court", "advocate",
            "nursing", "medical", "healthcare", "clinical", "agriculture", "farming",
            # General Soft Skills
            "communication", "teamwork", "analytical", "troubleshooting"
        }
        degree_terms = {
            "bachelor", "master", "phd", "doctorate", "b.s", "m.s", "b.sc", "m.sc", 
            "b.a", "m.a", "mba", "b.com", "m.com", "b.e", "b.tech", "m.tech", 
            "llb", "llm", "b.arch", "ca", "cfa", "cpa", "diploma"
        }
        edu_keywords = {"university", "college", "institute", "degree", "education"}
        cert_terms = {"certified", "certification", "certificate"}
        proj_terms = {"project", "projects", "built", "developed", "implemented", "created", "designed"}
        ach_terms = {"award", "winner", "recognition", "published", "patent", "honors", "achieved", "champion", "scholarship", "medal"}
        exp_terms = {
            "experience", "years", "senior", "lead", "manager", "engineer", "developer", "intern",
            "associate", "consultant", "executive", "officer", "specialist", "coordinator", 
            "director", "architect", "analyst", "supervisor", "administrator", "accountant", "advocate"
        }

        # Regex for stronger signals
        email_pattern = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
        phone_pattern = re.compile(r"\+?\d[\d\s\-()]{7,}\d")

        # Phrase detection for multi-word skills
        phrase_skills = [
            ("machine", "learning"), ("deep", "learning"), ("data", "science"),
            ("computer", "vision"), ("natural", "language", "processing"),
            ("node", "js"), ("next", "js"), ("react", "native"), ("web", "developer"),
            ("digital", "marketing"), ("human", "resources"), ("business", "analyst"),
            ("quality", "assurance"), ("customer", "service"), ("supply", "chain"),
            ("project", "management"), ("financial", "analysis"), ("civil", "engineer"),
            ("mechanical", "engineer"), ("electrical", "engineer"), ("software", "engineer")
        ]

        def set_label(i, base, is_start):
            labels[i] = f"B-{base}" if is_start else f"I-{base}"

        i = 0
        while i < len(tokens):
            tok_lower = tokens[i].lower()
            
            # Phrase Matching
            matched = False
            for phrase in phrase_skills:
                phrase_lower = [p.lower() for p in phrase]
                if [t.lower() for t in tokens[i:i+len(phrase)]] == phrase_lower:
                    for j in range(len(phrase)):
                        set_label(i + j, "SKILL", j == 0)
                    i += len(phrase)
                    matched = True
                    break
            if matched:
                continue

            # Single Token Logic
            if tok_lower in skill_terms:
                set_label(i, "SKILL", True)
            elif tok_lower in degree_terms or tok_lower in edu_keywords:
                set_label(i, "EDU", True)
            elif tok_lower in cert_terms:
                set_label(i, "CERT", True)
            elif tok_lower in proj_terms:
                set_label(i, "PROJ", True)
            elif tok_lower in ach_terms:
                set_label(i, "ACH", True)
            elif tok_lower in exp_terms:
                set_label(i, "EXP", True)
            elif email_pattern.fullmatch(tokens[i]):
                set_label(i, "CONTACT", True)
            elif phone_pattern.fullmatch(tokens[i]):
                set_label(i, "CONTACT", True)
            # Simple name check (heuristic)
            elif i < 2 and tokens[i].istitle() and tokens[i+1].istitle():
                set_label(i, "NAME", True)
                if i+1 < len(tokens): set_label(i+1, "NAME", False)
                i += 1

            i += 1

        return (tokens, labels)
    
    # Load TXT files
    for filepath in glob.glob(os.path.join(data_path, "**", "*.txt"), recursive=True):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            result = weak_label_text(text)
            if result:
                examples.append(result)
        except Exception:
            pass
    
    # Load CSV files
    try:
        import pandas as pd
        for filepath in glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True):
            try:
                df = pd.read_csv(filepath)
                text_col = None
                for col in ["resume", "resume_str", "text", "cv", "content"]:
                    if col.lower() in [c.lower() for c in df.columns]:
                        text_col = col
                        break
                if text_col:
                    for text in df[text_col].dropna().astype(str):
                        result = weak_label_text(text)
                        if result:
                            examples.append(result)
            except Exception:
                pass
    except ImportError:
        print("pandas not installed, skipping CSV files")
    
    # Load PDF files (requires pdfplumber)
    # A more robust PDF loading pipeline with OCR fallback
    try:
        import pdfplumber
        import fitz  # PyMuPDF
        from PIL import Image
        import numpy as np

        try:
            import easyocr
            print("\nInitializing EasyOCR (this may take a moment to load weights)...")
            ocr_reader = easyocr.Reader(['en'], gpu=True)
            ocr_available = True
        except ImportError:
            print("\n[!] EasyOCR not installed. Run 'pip install easyocr' for OCR support.")
            ocr_reader = None
            ocr_available = False

        pdf_files = glob.glob(os.path.join(data_path, "**", "*.pdf"), recursive=True)
        
        stats = {"total": len(pdf_files), "text": 0, "ocr": 0, "skipped": 0, "errors": 0}

        for filepath in tqdm(pdf_files, desc="Processing PDFs"):
            text = ""
            err_msgs = []
            
            # Check if it's actually a PDF file
            try:
                with open(filepath, 'rb') as f:
                    header = f.read(5)
                    if header != b'%PDF-':
                        stats["errors"] += 1
                        if stats["errors"] <= 5:
                            print(f"\nWarning: {os.path.basename(filepath)} is not a valid PDF (header: {header}).")
                            print("Hint: Kaggle download might have failed or downloaded HTML login pages.")
                        continue
            except Exception:
                pass

            # 1. Try PyMuPDF (fitz) native extraction first (much faster)
            try:
                doc = fitz.open(filepath)
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        text += page_text + "\n"
                text = text.strip()
                if text:
                    stats["text"] += 1
            except Exception as e:
                err_msgs.append(f"fitz error: {e}")

            # 2. Try pdfplumber if PyMuPDF returned empty
            if not text:
                try:
                    with pdfplumber.open(filepath) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    text = text.strip()
                    if text:
                        stats["text"] += 1
                except Exception as e:
                    err_msgs.append(f"pdfplumber error: {e}")

            # 3. Fallback to OCR if no text was extracted
            if not text and ocr_available:
                try:
                    doc = fitz.open(filepath)
                    ocr_text = ""
                    for page_num in range(min(3, len(doc))):
                        page = doc[page_num]
                        pix = page.get_pixmap(dpi=150, alpha=False)
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
                        result = ocr_reader.readtext(img_array, detail=0, paragraph=True)
                        ocr_text += " ".join(result) + "\n"
                    text = ocr_text.strip()
                    if text:
                        stats["ocr"] += 1
                except Exception as e:
                    err_msgs.append(f"OCR error: {e}")
                    text = "" # OCR failed, text remains empty

            # 3. Process the extracted text
            if text:
                result = weak_label_text(text)
                if result:
                    examples.append(result)
                else:
                    stats["skipped"] += 1
            else:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    print(f"\nWarning: Empty text from {os.path.basename(filepath)}. Errors: {' | '.join(err_msgs)}")

        print(f"\n--- PDF Loading Stats ---\nTotal: {stats['total']}, Text: {stats['text']}, OCR: {stats['ocr']}, Skipped (too short): {stats['skipped']}, Errors/Empty: {stats['errors']}\n-------------------------")

    except ImportError as e:
        print(f"\nWarning: Missing dependency for PDF processing: '{e.name}'. Please run 'pip install {e.name}'.")
        print("Hint: You will need 'PyMuPDF' and 'easyocr' via pip.")
        print("Run: pip install PyMuPDF easyocr pillow numpy")
    
    print(f"Loaded {len(examples)} resume examples")
    return examples


def generate_synthetic_data(num_examples: int = 2000):
    """Generate synthetic training data"""
    import random
    
    skills = ["Python", "JavaScript", "React", "Node.js", "TypeScript", "Java",
              "C++", "SQL", "MongoDB", "PostgreSQL", "AWS", "Docker", "Kubernetes",
              "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "NLP"]
    companies = ["Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix"]
    titles = ["Software Engineer", "Senior Developer", "Data Scientist", "ML Engineer"]
    universities = ["MIT", "Stanford", "UC Berkeley", "Carnegie Mellon"]
    degrees = ["Bachelor of Science in Computer Science", "Master of Science in Data Science"]
    
    examples = []
    for _ in range(num_examples):
        tokens, labels = [], []
        
        # Name
        tokens.extend(["John", "Doe"])
        labels.extend(["B-NAME", "I-NAME"])
        tokens.append("|")
        labels.append("O")
        
        # Skills
        tokens.extend(["Skills", ":"])
        labels.extend(["O", "O"])
        for i, skill in enumerate(random.sample(skills, random.randint(3, 6))):
            for j, s in enumerate(skill.split()):
                tokens.append(s)
                labels.append("B-SKILL" if j == 0 else "I-SKILL")
            if i < 5:
                tokens.append(",")
                labels.append("O")
        
        # Experience
        tokens.extend(["Experience", ":"])
        labels.extend(["O", "O"])
        title = random.choice(titles)
        for j, t in enumerate(title.split()):
            tokens.append(t)
            labels.append("B-EXP" if j == 0 else "I-EXP")
        tokens.append("at")
        labels.append("O")
        tokens.append(random.choice(companies))
        labels.append("B-ORG")
        
        # Education
        tokens.extend(["Education", ":"])
        labels.extend(["O", "O"])
        degree = random.choice(degrees)
        for j, d in enumerate(degree.split()):
            tokens.append(d)
            labels.append("B-EDU" if j == 0 else "I-EDU")
        
        examples.append((tokens, labels))
    
    return examples


# =====================
# Training
# =====================
def evaluate(model, dataloader, device):
    """Evaluate model on dataloader"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            total_loss += outputs["loss"].item()
            
            for pred_seq, label_seq, mask in zip(
                outputs["predictions"], labels.cpu().numpy(), attention_mask.cpu().numpy()
            ):
                pred_labels, true_labels = [], []
                pred_idx = 0
                for label, m in zip(label_seq, mask):
                    if m == 1 and label != -100:
                        if pred_idx < len(pred_seq):
                            pred_labels.append(ID2LABEL[pred_seq[pred_idx]])
                        else:
                            pred_labels.append("O")
                        true_labels.append(ID2LABEL[label])
                        pred_idx += 1
                if pred_labels:
                    all_preds.append(pred_labels)
                    all_labels.append(true_labels)
    
    return {
        "loss": total_loss / len(dataloader) if dataloader else 0,
        "f1": f1_score(all_labels, all_preds) if all_labels else 0,
        "preds": all_preds,
        "labels": all_labels
    }


def train(args):
    """Main training function"""
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("HireSense AI - Resume NER Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Platform: {platform.system()}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    # Load data
    print("\nLoading data...")
    if args.data_path and os.path.exists(args.data_path):
        all_examples = load_resume_data(args.data_path)
    else:
        print("No data path provided or path doesn't exist, using synthetic data")
        all_examples = []
    
    if len(all_examples) == 0:
        raise RuntimeError("0 resumes were successfully loaded. PDF extraction failed entirely. "
                           "Please ensure PyMuPDF is installed (pip install PyMuPDF).")
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_examples, temp = train_test_split(all_examples, test_size=0.2, random_state=42)
    val_examples, test_examples = train_test_split(temp, test_size=0.5, random_state=42)
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
    
    # Create datasets
    train_dataset = ResumeNERDataset(train_examples, tokenizer, args.max_length)
    val_dataset = ResumeNERDataset(val_examples, tokenizer, args.max_length)
    test_dataset = ResumeNERDataset(test_examples, tokenizer, args.max_length)
    
    # DataLoaders (num_workers=0 for Windows)
    num_workers = 0 if platform.system() == "Windows" else 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, 
                             num_workers=num_workers, pin_memory=True)
    
    # Create model
    print("\nInitializing model...")
    model = BertBiLSTMCRF(
        bert_model_name="bert-base-uncased",
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=2,
        lstm_dropout=0.3,
        freeze_bert=args.freeze_bert
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    bert_params = [p for n, p in model.named_parameters() if 'bert' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'bert' not in n and p.requires_grad]
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': args.bert_lr},
        {'params': other_params, 'lr': args.lstm_lr}
    ], weight_decay=0.01)
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * 0.1), 
        num_training_steps=num_training_steps
    )
    
    # Mixed precision
    scaler = GradScaler() if args.fp16 and torch.cuda.is_available() else None
    use_amp = args.fp16 and torch.cuda.is_available()
    
    print(f"\nMixed Precision (FP16): {use_amp}")
    print(f"Gradient Accumulation: {args.accumulation_steps}")
    print(f"Effective Batch Size: {args.batch_size * args.accumulation_steps}")
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    
    best_f1 = 0
    patience = 0
    history = []
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs["loss"] / args.accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    if scale_before <= scale_after:
                        scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"] / args.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * args.accumulation_steps
            
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                progress.set_postfix({"loss": f"{loss.item()*args.accumulation_steps:.4f}", 
                                      "gpu": f"{gpu_mem:.1f}GB"})
            else:
                progress.set_postfix({"loss": f"{loss.item()*args.accumulation_steps:.4f}"})
        
        # Evaluate
        val_results = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        print(f"\nEpoch {epoch}: Train Loss={avg_loss:.4f}, Val Loss={val_results['loss']:.4f}, Val F1={val_results['f1']:.4f}")
        
        history.append({
            "epoch": epoch, 
            "train_loss": avg_loss, 
            "val_loss": val_results["loss"], 
            "val_f1": val_results["f1"]
        })
        
        # Save best model
        if val_results["f1"] > best_f1:
            best_f1 = float(val_results["f1"])  # Convert numpy float to standard python float
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "f1": best_f1
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> Saved best model (F1: {best_f1:.4f})")
        else:
            patience += 1
        
        if patience >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_results = evaluate(model, test_loader, device)
    print(f"Test F1: {test_results['f1']:.4f}")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_results["labels"], test_results["preds"]))
    
    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "f1": test_results["f1"]
    }, os.path.join(args.output_dir, "resume_ner_model.pt"))
    
    # Save config
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump({
            "bert_model_name": "bert-base-uncased",
            "lstm_hidden_size": args.lstm_hidden,
            "lstm_num_layers": 2,
            "num_labels": NUM_LABELS,
            "label2id": LABEL2ID,
            "id2label": {str(k): v for k, v in ID2LABEL.items()}
        }, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Resume NER model")
    parser.add_argument("--data_path", type=str, default="./data/kaggle_resume_pdf",
                        help="Path to resume data directory")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size (8 recommended for 8GB VRAM)")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--lstm_hidden", type=int, default=256,
                        help="LSTM hidden size")
    parser.add_argument("--bert_lr", type=float, default=2e-5,
                        help="BERT learning rate")
    parser.add_argument("--lstm_lr", type=float, default=1e-3,
                        help="LSTM/CRF learning rate")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--freeze_bert", action="store_true", default=False,
                        help="Freeze BERT layers")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    train(args)

"""
HireSense AI - Dataset Loading and Preprocessing
Combines Resume Corpus + NER-Annotated-CVs datasets
"""

import os
import json
import re
import glob
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from config import (
    DataConfig, TrainingConfig, 
    LABEL2ID, ID2LABEL, ENTITY_LABELS
)


@dataclass
class NERExample:
    """Single NER training example"""
    tokens: List[str]
    labels: List[str]
    text: str = ""


class ResumeNERDataset(Dataset):
    """PyTorch Dataset for Resume NER"""
    
    def __init__(
        self,
        examples: List[NERExample],
        tokenizer: BertTokenizerFast,
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = LABEL2ID
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize with word-level alignment
        encoding = self.tokenizer(
            example.tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Align labels with subword tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens [CLS], [SEP], [PAD]
                aligned_labels.append(-100)  # Ignored in loss
            elif word_idx != previous_word_idx:
                # First subword of a word
                if word_idx < len(example.labels):
                    aligned_labels.append(self.label2id[example.labels[word_idx]])
                else:
                    aligned_labels.append(self.label2id["O"])
            else:
                # Subsequent subwords - use I- tag if original was B-
                if word_idx < len(example.labels):
                    label = example.labels[word_idx]
                    if label.startswith("B-"):
                        label = "I-" + label[2:]
                    aligned_labels.append(self.label2id.get(label, self.label2id["O"]))
                else:
                    aligned_labels.append(self.label2id["O"])
            previous_word_idx = word_idx
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }


class DataProcessor:
    """Process and combine multiple resume datasets"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def load_resume_corpus(self) -> List[NERExample]:
        """
        Load Resume Corpus dataset (36 entity types)
        Expected format: JSON with tokens and NER labels
        """
        examples = []
        corpus_path = self.config.resume_corpus_path
        
        if not os.path.exists(corpus_path):
            print(f"Warning: Resume Corpus not found at {corpus_path}")
            return examples
            
        # Load JSON files
        for filename in tqdm(os.listdir(corpus_path), desc="Loading Resume Corpus"):
            if filename.endswith(".json"):
                filepath = os.path.join(corpus_path, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Map to our entity schema
                tokens = data.get("tokens", [])
                labels = self._map_labels(data.get("labels", []))
                
                if tokens and labels and len(tokens) == len(labels):
                    examples.append(NERExample(
                        tokens=tokens,
                        labels=labels,
                        text=" ".join(tokens)
                    ))
                    
        return examples
    
    def load_ner_annotated_cvs(self) -> List[NERExample]:
        """
        Load NER-Annotated-CVs dataset (IT skills focused)
        Expected format: CoNLL-style or JSON
        """
        examples = []
        ner_path = self.config.ner_annotated_path
        
        if not os.path.exists(ner_path):
            print(f"Warning: NER-Annotated-CVs not found at {ner_path}")
            return examples
            
        # Try loading as CoNLL format
        conll_file = os.path.join(ner_path, "train.txt")
        if os.path.exists(conll_file):
            examples.extend(self._load_conll(conll_file))
            
        # Try loading as JSON
        json_file = os.path.join(ner_path, "annotations.json")
        if os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in tqdm(data, desc="Loading NER-Annotated-CVs"):
                    tokens = item.get("tokens", [])
                    labels = self._map_labels(item.get("ner_tags", []))
                    if tokens and labels:
                        examples.append(NERExample(
                            tokens=tokens,
                            labels=labels,
                            text=" ".join(tokens)
                        ))
                        
        return examples


    def load_kaggle_resume_pdf(self) -> List[NERExample]:
        """
        Load resumes from Kaggle PDF dataset with:
        - PDF parsing (pdfplumber)
        - OCR fallback (pytesseract)
        - TXT + CSV support
        """
        examples = []
        kaggle_path = self.config.kaggle_pdf_path

        if not os.path.exists(kaggle_path):
            print(f"Warning: Kaggle resume path not found at {kaggle_path}")
            return examples

        # =======================
        # LOAD PDF FILES
        # =======================
        # Use os.walk for better Windows compatibility
        pdf_files = []
        for root, dirs, files in os.walk(kaggle_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        print(f"Found {len(pdf_files)} PDF files in {kaggle_path}")
        
        # Track statistics for debugging
        stats = {
            "total": len(pdf_files),
            "pdfplumber_success": 0,
            "ocr_success": 0,
            "empty_text": 0,
            "too_short": 0,
            "errors": 0
        }

        for filepath in tqdm(pdf_files, desc="Loading PDF resumes"):
            try:
                text = ""

                # ---- Try normal extraction with pdfplumber ----
                try:
                    with pdfplumber.open(filepath) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    text = text.strip()
                    if text:
                        stats["pdfplumber_success"] += 1
                except Exception as e:
                    pass  # Will try OCR next
                except (FileNotFoundError, NameError) as e:
                    print("Could not find poppler, install it with brew install poppler")
                    raise e

                 # ---- OCR fallback if pdfplumber returned empty ----
                if not text:
                    try:
                        doc = fitz.open(filepath)
                        ocr_text = ""
                        for page_num in range(min(3, len(doc))):
                            page = doc[page_num]
                            pix = page.get_pixmap(dpi=150)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            ocr_text += pytesseract.image_to_string(img) + "\n"
                        text = ocr_text.strip()
                        if text:
                            stats["ocr_success"] += 1
                    except Exception as ocr_error:
                        pass

                if not text:
                    stats["empty_text"] += 1
                    continue

                example = self._weak_label_resume_text(text)

                if example is None or len(example.tokens) <= 5:
                    stats["too_short"] += 1
                    continue
                    
                examples.append(example)

            except Exception as e:
                stats["errors"] += 1
                # Only print first few errors to avoid spam
                if stats["errors"] <= 5:
                    print(f"Warning: failed reading PDF {os.path.basename(filepath)}: {e}")
        
        # Print statistics
        print(f"\n--- PDF Loading Statistics ---")
        print(f"Total PDFs found: {stats['total']}")
        print(f"pdfplumber extracted: {stats['pdfplumber_success']}")
        print(f"OCR extracted: {stats['ocr_success']}")
        print(f"Empty (no text): {stats['empty_text']}")
        print(f"Too short (<=5 tokens): {stats['too_short']}")
        print(f"Errors: {stats['errors']}")
        print(f"Successfully loaded: {len(examples)}")
        print(f"------------------------------\n")

        print(f"Total usable resumes loaded: {len(examples)}")

        return examples

    def _weak_label_resume_text(self, text: str) -> Optional[NERExample]:
        """Create improved weak BIO labels from raw resume text."""
        import re

        # 🔥 Guard 1: remove garbage text
        if not text or len(text.strip()) < 20:
            return None

        tokens = re.findall(r"\b\w+[\w+.+-]*\b|[^\w\s]", text)

        # 🔥 Guard 2: too small = useless
        if len(tokens) < 5:
            return None

        labels = ["O"] * len(tokens)

        # 🔥 Better vocab (expanded)
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
            "llb", "llm", "b.arch", "ca", "cfa", "cpa", "diploma", "university", "college"
        }

        cert_terms = {"certified", "certification", "certificate", "aws", "azure", "gcp"}

        proj_terms = {"project", "projects", "built", "developed", "implemented"}

        ach_terms = {"award", "winner", "recognition", "published", "patent", "honors", "achieved", "champion", "scholarship", "medal"}

        exp_terms = {
            "experience", "years", "senior", "lead", "manager", "engineer", "developer", "intern",
            "associate", "consultant", "executive", "officer", "specialist", "coordinator", 
            "director", "architect", "analyst", "supervisor", "administrator", "accountant", "advocate"
        }

        # 🔥 Regex patterns (much stronger signal)
        email_pattern = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
        phone_pattern = re.compile(r"\+?\d[\d\s\-]{7,}\d")

        # 🔥 Phrase detection (important upgrade)
        phrase_skills = [
            ["machine", "learning"],
            ["deep", "learning"],
            ["data", "science"],
            ["computer", "vision"],
            ["natural", "language", "processing"],
            ["node", "js"], ["next", "js"], ["react", "native"], ["web", "developer"],
            ["digital", "marketing"], ["human", "resources"], ["business", "analyst"],
            ["quality", "assurance"], ["customer", "service"], ["supply", "chain"],
            ["project", "management"], ["financial", "analysis"], ["civil", "engineer"],
            ["mechanical", "engineer"], ["electrical", "engineer"], ["software", "engineer"]
        ]

        # 🔥 Helper
        def set_label(i, base, is_start):
            labels[i] = f"B-{base}" if is_start else f"I-{base}"

        i = 0
        while i < len(tokens):
            tok = tokens[i].lower()

            # =========================
            # 🔥 Phrase matching
            # =========================
            matched = False
            for phrase in phrase_skills:
                if tokens[i:i+len(phrase)] == phrase:
                    for j in range(len(phrase)):
                        set_label(i + j, "SKILL", j == 0)
                    i += len(phrase)
                    matched = True
                    break
            if matched:
                continue

            # =========================
            # 🔥 Single token logic
            # =========================
            if tok in skill_terms:
                set_label(i, "SKILL", True)

            elif tok in degree_terms:
                set_label(i, "EDU", True)

            elif tok in cert_terms:
                set_label(i, "CERT", True)

            elif tok in proj_terms:
                set_label(i, "PROJ", True)

            elif tok in ach_terms:
                set_label(i, "ACH", True)

            elif tok in exp_terms or re.fullmatch(r"\d+", tok):
                set_label(i, "EXP", True)

            elif email_pattern.fullmatch(tokens[i]):
                set_label(i, "CONTACT", True)

            elif phone_pattern.fullmatch(tokens[i]):
                set_label(i, "CONTACT", True)

            i += 1

        return NERExample(tokens=tokens, labels=labels, text=" ".join(tokens))
    
    def _load_conll(self, filepath: str) -> List[NERExample]:
        """Load CoNLL format file"""
        examples = []
        tokens, labels = [], []
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if tokens:
                        examples.append(NERExample(
                            tokens=tokens,
                            labels=self._map_labels(labels),
                            text=" ".join(tokens)
                        ))
                    tokens, labels = [], []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        labels.append(parts[-1])
                        
        # Don't forget last example
        if tokens:
            examples.append(NERExample(
                tokens=tokens,
                labels=self._map_labels(labels),
                text=" ".join(tokens)
            ))
            
        return examples
    
    def _map_labels(self, labels: List[str]) -> List[str]:
        """Map dataset-specific labels to our unified schema"""
        # Mapping from various dataset labels to our schema
        label_mapping = {
            # Skills
            "B-Skills": "B-SKILL", "I-Skills": "I-SKILL",
            "B-SKILLS": "B-SKILL", "I-SKILLS": "I-SKILL",
            "B-Skill": "B-SKILL", "I-Skill": "I-SKILL",
            "B-TECHNOLOGY": "B-SKILL", "I-TECHNOLOGY": "I-SKILL",
            "B-PROGRAMMING": "B-SKILL", "I-PROGRAMMING": "I-SKILL",
            
            # Experience
            "B-Experience": "B-EXP", "I-Experience": "I-EXP",
            "B-EXPERIENCE": "B-EXP", "I-EXPERIENCE": "I-EXP",
            "B-WORK_EXPERIENCE": "B-EXP", "I-WORK_EXPERIENCE": "I-EXP",
            "B-JOB_TITLE": "B-EXP", "I-JOB_TITLE": "I-EXP",
            "B-DESIGNATION": "B-EXP", "I-DESIGNATION": "I-EXP",
            
            # Education
            "B-Education": "B-EDU", "I-Education": "I-EDU",
            "B-EDUCATION": "B-EDU", "I-EDUCATION": "I-EDU",
            "B-DEGREE": "B-EDU", "I-DEGREE": "I-EDU",
            "B-COLLEGE": "B-EDU", "I-COLLEGE": "I-EDU",
            "B-UNIVERSITY": "B-EDU", "I-UNIVERSITY": "I-EDU",
            
            # Organizations
            "B-Company": "B-ORG", "I-Company": "I-ORG",
            "B-COMPANY": "B-ORG", "I-COMPANY": "I-ORG",
            "B-ORGANIZATION": "B-ORG", "I-ORGANIZATION": "I-ORG",
            "B-ORG": "B-ORG", "I-ORG": "I-ORG",
            
            # Projects
            "B-Project": "B-PROJ", "I-Project": "I-PROJ",
            "B-PROJECTS": "B-PROJ", "I-PROJECTS": "I-PROJ",
            
            # Achievements/Certifications
            "B-Certification": "B-CERT", "I-Certification": "I-CERT",
            "B-CERTIFICATION": "B-CERT", "I-CERTIFICATION": "I-CERT",
            "B-Achievement": "B-ACH", "I-Achievement": "I-ACH",
            "B-AWARD": "B-ACH", "I-AWARD": "I-ACH",
            
            # Location
            "B-Location": "B-LOC", "I-Location": "I-LOC",
            "B-LOCATION": "B-LOC", "I-LOCATION": "I-LOC",
            "B-CITY": "B-LOC", "I-CITY": "I-LOC",
            
            # Dates
            "B-Date": "B-DATE", "I-Date": "I-DATE",
            "B-DATE": "B-DATE", "I-DATE": "I-DATE",
            "B-YEAR": "B-DATE", "I-YEAR": "I-DATE",
            
            # Name
            "B-Name": "B-NAME", "I-Name": "I-NAME",
            "B-NAME": "B-NAME", "I-NAME": "I-NAME",
            
            # Contact
            "B-Email": "B-CONTACT", "I-Email": "I-CONTACT",
            "B-EMAIL": "B-CONTACT", "I-EMAIL": "I-CONTACT",
            "B-Phone": "B-CONTACT", "I-Phone": "I-CONTACT",
            "B-PHONE": "B-CONTACT", "I-PHONE": "I-CONTACT",
        }
        
        mapped = []
        for label in labels:
            mapped_label = label_mapping.get(label, label)
            # If not in our schema, default to O
            if mapped_label not in LABEL2ID:
                mapped_label = "O"
            mapped.append(mapped_label)
            
        return mapped
    
    def create_synthetic_examples(self, num_examples: int = 1000) -> List[NERExample]:
        """
        Create synthetic training examples for bootstrapping
        when real data is not available
        """
        import random
        
        # Sample resume components
        skills = [
            "Python", "JavaScript", "React", "Node.js", "TypeScript", "Java",
            "C++", "SQL", "MongoDB", "PostgreSQL", "AWS", "Docker", "Kubernetes",
            "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "NLP",
            "Git", "Linux", "REST API", "GraphQL", "Redis", "Elasticsearch"
        ]
        
        companies = [
            "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix",
            "Uber", "Airbnb", "Stripe", "Spotify", "Twitter", "LinkedIn"
        ]
        
        titles = [
            "Software Engineer", "Senior Developer", "Data Scientist",
            "Machine Learning Engineer", "Full Stack Developer", "DevOps Engineer",
            "Technical Lead", "Engineering Manager", "Product Manager"
        ]
        
        universities = [
            "MIT", "Stanford University", "UC Berkeley", "Carnegie Mellon",
            "Georgia Tech", "University of Michigan", "Harvard University"
        ]
        
        degrees = [
            "Bachelor of Science in Computer Science",
            "Master of Science in Data Science",
            "PhD in Machine Learning",
            "Bachelor of Engineering"
        ]
        
        examples = []
        
        for _ in range(num_examples):
            tokens = []
            labels = []
            
            # Add name
            tokens.extend(["John", "Doe"])
            labels.extend(["B-NAME", "I-NAME"])
            tokens.append("|")
            labels.append("O")
            
            # Add skills section
            tokens.append("Skills")
            labels.append("O")
            tokens.append(":")
            labels.append("O")
            
            selected_skills = random.sample(skills, random.randint(3, 8))
            for i, skill in enumerate(selected_skills):
                skill_tokens = skill.split()
                for j, st in enumerate(skill_tokens):
                    tokens.append(st)
                    labels.append("B-SKILL" if j == 0 else "I-SKILL")
                if i < len(selected_skills) - 1:
                    tokens.append(",")
                    labels.append("O")
            
            tokens.append(".")
            labels.append("O")
            
            # Add experience section
            tokens.append("Experience")
            labels.append("O")
            tokens.append(":")
            labels.append("O")
            
            title = random.choice(titles)
            for j, t in enumerate(title.split()):
                tokens.append(t)
                labels.append("B-EXP" if j == 0 else "I-EXP")
            
            tokens.append("at")
            labels.append("O")
            
            company = random.choice(companies)
            tokens.append(company)
            labels.append("B-ORG")
            
            tokens.append("(")
            labels.append("O")
            tokens.append(f"{random.randint(2015, 2023)}")
            labels.append("B-DATE")
            tokens.append("-")
            labels.append("O")
            tokens.append("Present")
            labels.append("B-DATE")
            tokens.append(")")
            labels.append("O")
            
            # Add education section
            tokens.append("Education")
            labels.append("O")
            tokens.append(":")
            labels.append("O")
            
            degree = random.choice(degrees)
            for j, d in enumerate(degree.split()):
                tokens.append(d)
                labels.append("B-EDU" if j == 0 else "I-EDU")
            
            tokens.append("from")
            labels.append("O")
            
            uni = random.choice(universities)
            for j, u in enumerate(uni.split()):
                tokens.append(u)
                labels.append("B-ORG" if j == 0 else "I-ORG")
            
            examples.append(NERExample(
                tokens=tokens,
                labels=labels,
                text=" ".join(tokens)
            ))
        
        return examples
    
    def load_all_data(self) -> Tuple[List[NERExample], List[NERExample], List[NERExample]]:
        """Load and combine all datasets, then split"""
        all_examples = []
        
        # Load real datasets
        resume_corpus = self.load_resume_corpus()
        print(f"Loaded {len(resume_corpus)} examples from Resume Corpus")
        all_examples.extend(resume_corpus)
        
        ner_annotated = self.load_ner_annotated_cvs()
        print(f"Loaded {len(ner_annotated)} examples from NER-Annotated-CVs")
        all_examples.extend(ner_annotated)

        kaggle_examples = self.load_kaggle_resume_pdf()
        print(f"Loaded {len(kaggle_examples)} weakly-labeled examples from Kaggle resume-data-pdf")
        all_examples.extend(kaggle_examples)
        
        # If no real data, use synthetic
        if len(all_examples) < 100:
            print("Insufficient real data, generating synthetic examples...")
            synthetic = self.create_synthetic_examples(2000)
            all_examples.extend(synthetic)
            print(f"Added {len(synthetic)} synthetic examples")
        
        print(f"Total examples: {len(all_examples)}")
        
        # Split data
        train_examples, temp_examples = train_test_split(
            all_examples,
            test_size=(self.config.val_ratio + self.config.test_ratio),
            random_state=42
        )
        
        val_examples, test_examples = train_test_split(
            temp_examples,
            test_size=self.config.test_ratio / (self.config.val_ratio + self.config.test_ratio),
            random_state=42
        )
        
        print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
        
        return train_examples, val_examples, test_examples


def create_dataloaders(
    train_examples: List[NERExample],
    val_examples: List[NERExample],
    test_examples: List[NERExample],
    tokenizer: BertTokenizerFast,
    train_config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders - Windows compatible"""
    import platform
    
    train_dataset = ResumeNERDataset(
        train_examples, tokenizer, train_config.max_seq_length
    )
    val_dataset = ResumeNERDataset(
        val_examples, tokenizer, train_config.max_seq_length
    )
    test_dataset = ResumeNERDataset(
        test_examples, tokenizer, train_config.max_seq_length
    )
    
    # Windows requires num_workers=0 due to multiprocessing issues
    # Linux/Mac can use multiple workers for faster data loading
    num_workers = 0 if platform.system() == "Windows" else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(train_config.device == "cuda"),
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(train_config.device == "cuda"),
        persistent_workers=(num_workers > 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(train_config.device == "cuda"),
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from transformers import BertTokenizerFast
    
    # Test data loading
    data_config = DataConfig()
    processor = DataProcessor(data_config)
    
    train, val, test = processor.load_all_data()
    
    # Test tokenization
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    train_config = TrainingConfig()
    train_loader, val_loader, test_loader = create_dataloaders(
        train, val, test, tokenizer, train_config
    )
    
    # Check a batch
    batch = next(iter(train_loader))
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")

"""
HireSense AI — Self-Contained Local Training Script
BERT + BiLSTM + CRF, Multi-Sector Resume NER

Usage:
    python train_local.py --data_path ./data/kaggle_resume_pdf --fp16

All logic is in one file (no imports from config/dataset/model).
"""

import os, re, json, glob, random, argparse, platform
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup
from torchcrf import CRF
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score


# =============================================================================
# Label schema
# =============================================================================

ENTITY_LABELS = [
    "O",
    "B-SKILL","I-SKILL","B-EXP","I-EXP","B-EDU","I-EDU",
    "B-PROJ","I-PROJ","B-ACH","I-ACH","B-CERT","I-CERT",
    "B-ORG","I-ORG","B-LOC","I-LOC","B-DATE","I-DATE",
    "B-NAME","I-NAME","B-CONTACT","I-CONTACT","B-SECTOR","I-SECTOR",
]
LABEL2ID  = {l: i for i, l in enumerate(ENTITY_LABELS)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(ENTITY_LABELS)

RESUME_SECTION_HEADERS = {
    "education","experience","skills","projects","achievements",
    "certifications","awards","summary","objective","profile","contact",
    "internships","positions","extracurricular","activities","interests",
    "startup","competitions","academic",
}


# =============================================================================
# Model
# =============================================================================

class BertBiLSTMCRF(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", lstm_hidden=256,
                 lstm_layers=2, lstm_dropout=0.3, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        if freeze_bert:
            for p in self.bert.parameters(): p.requires_grad = False
        h = self.bert.config.hidden_size
        self.lstm = nn.LSTM(h, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=lstm_dropout if lstm_layers>1 else 0)
        self.dropout    = nn.Dropout(lstm_dropout)
        self.hidden2tag = nn.Linear(lstm_hidden*2, NUM_LABELS)
        self.crf        = CRF(NUM_LABELS, batch_first=True)
        self._init()

    def _init(self):
        for n, p in self.lstm.named_parameters():
            if "weight_ih" in n: nn.init.xavier_uniform_(p)
            elif "weight_hh" in n: nn.init.orthogonal_(p)
            elif "bias" in n:
                nn.init.zeros_(p)
                sz = p.size(0); p.data[sz//4:sz//2].fill_(1.0)
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)

    def forward(self, ids, mask, labels=None):
        seq  = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        seq,_= self.lstm(seq)
        seq  = self.dropout(seq)
        emit = self.hidden2tag(seq).float()
        m    = mask.bool()
        out  = {"emissions": emit}
        if labels is not None:
            lc = labels.clone(); lc[labels==-100] = 0
            out["loss"] = -self.crf(emit, lc, mask=m, reduction="mean")
        with torch.no_grad():
            out["predictions"] = self.crf.decode(emit, mask=m)
        return out


# =============================================================================
# Weak labeller (self-contained copy, same logic as dataset.py)
# =============================================================================

SKILL_SINGLE = {
    "python","java","c++","c#","c","javascript","typescript","ruby","php",
    "go","golang","rust","scala","kotlin","swift","r","bash","shell",
    "react","reactjs","angular","vue","nextjs","tailwind","bootstrap","sass",
    "node","nodejs","express","django","flask","fastapi","spring","nestjs",
    "flutter","dart","mysql","postgresql","postgres","mongodb","sqlite","redis",
    "firebase","supabase","mssql","dynamodb","elasticsearch",
    "pytorch","tensorflow","keras","sklearn","xgboost","numpy","pandas",
    "matplotlib","seaborn","nltk","spacy","huggingface","bert","yolo","yolov8",
    "langchain","openai","gemini","rag","pennylane","opencv","databricks",
    "redshift","spark","hadoop","kafka","airflow","dbt","etl","tableau","sas",
    "aws","azure","gcp","heroku","netlify","vercel","docker","kubernetes","k8s",
    "jenkins","terraform","ansible","linux","ubuntu","git","github","gitlab",
    "kali","wireshark","metasploit","nmap","splunk","bloomberg","tally","excel",
    "powerbi","dsa","oop","dbms","jwt","oauth","agile","scrum","rest","graphql",
    "figma","sketch","canva","photoshop","autocad","revit","solidworks","matlab",
    "plc","scada","communication","teamwork","leadership","analytical",
}
DEGREE_SINGLE  = {
    "bachelor","master","phd","doctorate","mba","btech","mtech","b.tech","m.tech",
    "b.sc","m.sc","b.com","m.com","b.e","m.e","b.a","m.a","llb","llm","mbbs",
    "md","pgdm","ca","cfa","cpa","diploma","b.s","m.s",
    "university","college","institute","iit","nit","bits","iiit",
}
CERT_SINGLE    = {"certified","certification","certificate","oracle","bloomberg",
                  "nptel","coursera","udemy","hackerrank","freecodecamp","scalar",
                  "cs50","pmp","cissp","ceh","comptia","databricks"}
PROJ_SINGLE    = {"project","projects","built","developed","implemented",
                  "created","designed","deployed","architected","engineered"}
ACH_SINGLE     = {"award","winner","finalist","recognition","published","patent",
                  "honors","scholarship","champion","medal","ranked","selected","hackathon"}
EXP_SINGLE     = {"senior","junior","lead","principal","associate","head",
                  "engineer","developer","analyst","scientist","architect","designer",
                  "manager","director","officer","specialist","intern","internship",
                  "founder","co-founder","president","consultant","trainee","freelancer"}
ORG_SINGLE     = {"google","microsoft","amazon","meta","apple","tcs","infosys","wipro",
                  "hcl","cognizant","capgemini","accenture","jio","reliance","flipkart",
                  "deloitte","pwc","ey","kpmg","hdfc","icici","nysonaiconnect","nsoft",
                  "hacktify","mactores","cognifyz","meryt","isf","gostops"}
LOC_SINGLE     = {"jabalpur","mumbai","pune","bangalore","bengaluru","delhi","noida",
                  "hyderabad","chennai","kolkata","vellore","jaipur","surat","india","mp"}

PHRASE_SKILLS = [
    ("machine","learning"),("deep","learning"),("natural","language","processing"),
    ("computer","vision"),("generative","ai"),("data","science"),("data","analysis"),
    ("data","engineering"),("google","cloud"),("google","cloud","platform"),
    ("microsoft","azure"),("aws","glue"),("github","actions"),
    ("full","stack"),("mern","stack"),("react","native"),("node","js"),("next","js"),
    ("rest","api"),("restful","api"),("penetration","testing"),("network","security"),
    ("financial","modelling"),("financial","analysis"),("investment","banking"),
    ("risk","management"),("algorithmic","trading"),("bloomberg","market","concepts"),
    ("bloomberg","finance","fundamentals"),("power","bi"),("apache","airflow"),
    ("problem","solving"),("critical","thinking"),("project","management"),
]
SENIORITY  = {"senior","junior","lead","principal","associate","staff","chief","head","founding"}
ROLE_WORDS = {"engineer","developer","analyst","scientist","architect","designer",
              "manager","director","officer","specialist","consultant","intern",
              "coordinator","executive","founder","co-founder","president","technician"}

ACH_PHRASE_RE = re.compile(
    r"\b(1st|2nd|3rd|\d+th)\s+(place|position|rank|prize)\b"
    r"|\btop\s+\d+[%+]?\b"
    r"|\b\d+\+\s*(dsa|problems|questions|teams|days)\b"
    r"|\b(gold|silver|bronze)\s+medal\b"
    r"|\bnational\s+(finalist|winner)\b|\bfinalist\b"
    r"|\b(won|secured|ranked)\s+(1st|2nd|first|second)\b",
    re.IGNORECASE,
)
CGPA_RE  = re.compile(r"\d+\.?\d*/\d+|\d+\.\d+\s*cgpa", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?91[\s\-]?\d{10}|\+?\d[\d\s\-()]{7,}\d")
DATE_RE  = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}"
    r"|\b\d{4}\s*[-\u2013\u2014]\s*(\d{4}|present|ongoing)\b|\b(20\d{2})\b",
    re.IGNORECASE,
)


def _build_token_offsets(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Compute (start_char, end_char) for each token by scanning text left-to-right.
    Returns (-1, -1) when a token cannot be located.
    """
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        idx = text.find(tok, pos)
        if idx == -1:
            offsets.append((-1, -1))
        else:
            offsets.append((idx, idx + len(tok)))
            pos = idx + 1
    return offsets


def weak_label(text: str):
    if not text or len(text.strip()) < 20:
        return None
    tokens = re.findall(
        r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|[\w]+(?:[./+\-#][\w]+)*|\S", text)
    if len(tokens) < 5:
        return None

    labels = ["O"] * len(tokens)
    tl     = [t.lower() for t in tokens]

    def safe(i: int, base: str, start: bool):
        """Assign label only if token still has 'O' (avoids overwriting)."""
        if 0 <= i < len(labels) and labels[i] == "O":
            labels[i] = f"B-{base}" if start else f"I-{base}"

    # Pre-compute character offsets for every token once (O(n) total)
    tok_offsets = _build_token_offsets(text, tokens)

    # ---- DATE  [BUG-5 FIX] ----
    # Original code: safe(ti, "DATE", True) — always B-DATE for every token.
    # Fix: first_in_span tracks B vs I correctly.
    for m in DATE_RE.finditer(text):
        first_in_span = True
        for ti, (s, _) in enumerate(tok_offsets):
            if s != -1 and m.start() <= s < m.end():
                safe(ti, "DATE", first_in_span)
                first_in_span = False

    # ---- CONTACT ----
    for ti, tok in enumerate(tokens):
        if EMAIL_RE.fullmatch(tok):
            safe(ti, "CONTACT", True)
        elif PHONE_RE.fullmatch(tok.replace(" ", "")):
            safe(ti, "CONTACT", True)

    # detect project lines
    if "project" in text.lower():
        for i, tok in enumerate(tl):
            if tok not in RESUME_SECTION_HEADERS and len(tok) > 3:
                safe(i, "PROJ", i == 0)

    if tok.endswith("pur") or tok.endswith("abad"):
        safe(i, "LOC", True)

    # ---- ACHIEVEMENT phrases ----
    for m in ACH_PHRASE_RE.finditer(text):
        span = m.group(0).lower().split()
        for ti in range(len(tokens) - len(span) + 1):
            if [tl[ti+k] for k in range(len(span))] == span:
                for k in range(len(span)):
                    safe(ti+k, "ACH", k == 0)

    # ---- PHRASE SKILLS ----
    i = 0
    while i < len(tokens):
        matched = False
        for phrase in PHRASE_SKILLS:
            end = i + len(phrase)
            if end <= len(tokens) and tuple(tl[i:end]) == phrase:
                for j in range(len(phrase)):
                    safe(i+j, "SKILL", j == 0)
                i += len(phrase); matched = True; break
        if not matched:
            i += 1

    # ---- Multi-word job titles ----
    i = 0
    while i < len(tokens):
        if tl[i] in SENIORITY and i+1 < len(tokens) and tl[i+1] in ROLE_WORDS:
            safe(i, "EXP", True); safe(i+1, "EXP", False)
            j = i + 2
            while j < len(tokens) and j < i+4 and tl[j] in ROLE_WORDS:
                safe(j, "EXP", False); j += 1
            i = j
        else:
            i += 1

    # ---- Single-token heuristics ----
    for i, tok in enumerate(tl):
        if tok in RESUME_SECTION_HEADERS: continue
        if tok in SKILL_SINGLE:    safe(i, "SKILL", True)
        elif tok in DEGREE_SINGLE: safe(i, "EDU",   True)
        elif tok in CERT_SINGLE:   safe(i, "CERT",  True)
        elif tok in PROJ_SINGLE:   safe(i, "PROJ",  True)
        elif tok in ACH_SINGLE:    safe(i, "ACH",   True)
        elif tok in EXP_SINGLE:    safe(i, "EXP",   True)
        elif tok in ORG_SINGLE:    safe(i, "ORG",   True)
        elif tok in LOC_SINGLE:    safe(i, "LOC",   True)
        elif CGPA_RE.match(tokens[i]): safe(i, "EDU", True)

    return (tokens, labels)


# =============================================================================
# Synthetic data (same templates as dataset.py)
# =============================================================================

import random

def generate_synthetic(n=2000):
    SECTOR_DATA = [
        {"sector":"cs_developer",
         "skills":[["Python","JavaScript","React","Node.js","MongoDB","Docker","AWS","JWT"],
                   ["Java","C++","SQL","Docker","Git","TypeScript","PostgreSQL","Flask"],
                   ["Python","FastAPI","MySQL","Redis","Linux","Docker","Firebase"]],
         "titles":["Full Stack Developer Intern","Software Engineer Intern",
                   "Product Engineer Intern","App Development Intern"],
         "orgs":["Google","Microsoft","TCS","Infosys","NysonAiConnect","Jio Platforms"],
         "degrees":["Bachelor of Technology in Information Technology",
                    "Bachelor of Technology in Computer Science"],
         "colleges":["Jabalpur Engineering College","Vellore Institute of Technology",
                     "NIT Trichy"],
         "ach":["Solved 500+ DSA problems LeetCode GeeksForGeeks",
                "Finalist HackCrux 2025 LNMIIT Jaipur","1st Place HackCrux 2025 90+ teams",
                "Smart India Hackathon 2024 Semifinalist"],
         "certs":["Oracle Cloud Infrastructure Architect Associate",
                  "Full-Stack Web Development MERN CodeHelp",
                  "Machine Learning with Python freecodecamp"]},
        # baaki same rehne do...
    ]

    FIRST = ["Priya","Rahul","Ananya","Arjun","Sneha","Divya","Karan","Neha",
             "Vikram","Rohit","Aditya","Sanidhya","Monal","Sonu","Jayendra",
             "Siddharth","Anshika","Arya","Shravan","Mansi","Ishan","Sanyam",
             "Kinjal","Vansh","Subrato","Sanskar","Samkit","Shreeansh"]
    LAST  = ["Sharma","Patel","Gupta","Singh","Jain","Tiwari","Agarwal","Kumar",
             "Narang","Sanghvi","Israni","Vishwakarma","Bhagat","Saxena","Tomar"]

    examples = []

    for _ in range(n):
        d = random.choice(SECTOR_DATA)
        t, l = [], []

        def add(words, tag):
            for j, w in enumerate(words):
                t.append(w)
                l.append(f"B-{tag}" if j==0 else f"I-{tag}")

        def o(w):
            t.append(w)
            l.append("O")

        # -------- NAME --------
        add(f"{random.choice(FIRST)} {random.choice(LAST)}".split(), "NAME")

        # -------- CONTACT (MESSY) --------
        if random.random() > 0.3:
            o("\nEmail:")
            o(f"{random.choice(FIRST).lower()}{random.randint(10,999)}@gmail.com")

        if random.random() > 0.5:
            o("Phone:")
            t.append(f"+91{random.randint(7000000000,9999999999)}")
            l.append("B-CONTACT")

        if random.random() > 0.6:
            o("LinkedIn:")
            o("linkedin.com/in/" + random.choice(FIRST).lower())

        # -------- RANDOM SECTION ORDER --------
        sections = ["skills","experience","education","projects","achievements","certs"]
        random.shuffle(sections)

        for sec in sections:

            # ===== SKILLS =====
            if sec == "skills":
                o("\n" + random.choice(["SKILLS","Skills","Technical Skills"]))
                sset = random.choice(d["skills"])

                for sk in random.sample(sset, min(random.randint(3,6), len(sset))):
                    if random.random() > 0.5:
                        o("•")
                    add(sk.split(), "SKILL")

            # ===== EXPERIENCE =====
            elif sec == "experience":
                o("\n" + random.choice(["Experience","WORK EXPERIENCE","Internships"]))

                for _ in range(random.randint(1,3)):
                    role = random.choice(d["titles"])
                    org = random.choice(d["orgs"])

                    if random.random() > 0.5:
                        add(role.split(), "EXP")
                        o("at")
                        add(org.split(), "ORG")
                    else:
                        add(org.split(), "ORG")
                        o("-")
                        add(role.split(), "EXP")

                    # dates (random format)
                    yr = random.randint(2020,2024)
                    o("(")
                    t.append(str(yr)); l.append("B-DATE")
                    o("-")
                    t.append(random.choice(["Present",str(yr+1)])); l.append("B-DATE")
                    o(")")

                    # description (noise)
                    o(random.choice([
                        "Worked on scalable systems",
                        "Built dashboards and pipelines",
                        "Collaborated with teams",
                        "Improved model performance"
                    ]))

            # ===== EDUCATION =====
            elif sec == "education":
                o("\n" + random.choice(["Education","ACADEMIC DETAILS"]))

                for _ in range(random.randint(1,2)):
                    add(random.choice(d["degrees"]).split(), "EDU")
                    o(",")
                    add(random.choice(d["colleges"]).split(), "ORG")

                    # CGPA noise
                    t.append(str(round(random.uniform(6.0,9.8),2)))
                    l.append("B-EDU")

            # ===== PROJECTS =====
            elif sec == "projects":
                o("\nPROJECTS")

                for _ in range(random.randint(1,3)):
                    o("•")
                    o(random.choice([
                        "AI Chatbot using NLP",
                        "Stock Prediction using LSTM",
                        "Portfolio Optimization System",
                        "Fraud Detection Model"
                    ]))

            # ===== ACHIEVEMENTS =====
            elif sec == "achievements":
                o("\nAchievements")
                add(random.choice(d["ach"]).split(), "ACH")

            # ===== CERTIFICATIONS =====
            elif sec == "certs":
                o("\nCertifications")
                add(random.choice(d["certs"]).split(), "CERT")

        # -------- LOCATION --------
        if random.random() > 0.5:
            o("|")
            t.append(random.choice(["Jabalpur","Mumbai","Pune","Bangalore","Delhi"]))
            l.append("B-LOC")

        examples.append((t, l))

    return examples


# =============================================================================
# Dataset
# =============================================================================

class ResumeNERDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=512):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        tokens, labels = self.examples[idx]
        enc = self.tokenizer(tokens, is_split_into_words=True,
                             max_length=self.max_len, padding="max_length",
                             truncation=True, return_tensors="pt")
        wids    = enc.word_ids()
        aligned = []
        prev    = None
        for wid in wids:
            if wid is None: aligned.append(-100)
            elif wid != prev:
                raw = labels[wid] if wid < len(labels) else "O"
                aligned.append(LABEL2ID.get(raw, 0))
            else:
                raw = labels[wid] if wid < len(labels) else "O"
                if raw.startswith("B-"): raw = "I-"+raw[2:]
                aligned.append(LABEL2ID.get(raw, 0))
            prev = wid
        return {"input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(aligned, dtype=torch.long)}


# =============================================================================
# Data loading
# =============================================================================

def load_resume_data(data_path):
    examples = []

    for fp in glob.glob(os.path.join(data_path,"**","*.txt"), recursive=True):
        try:
            with open(fp,"r",encoding="utf-8",errors="ignore") as f: text=f.read()
            r = weak_label(text)
            if r: examples.append(r)
        except: pass

    try:
        import pandas as pd
        for fp in glob.glob(os.path.join(data_path,"**","*.csv"), recursive=True):
            try:
                df = pd.read_csv(fp)
                col = next((c for c in df.columns if c.lower() in
                            {"resume","resume_str","text","cv","content"}), None)
                if col:
                    for txt in df[col].dropna().astype(str):
                        r = weak_label(txt)
                        if r: examples.append(r)
            except: pass
    except ImportError: pass

    pdf_files = glob.glob(os.path.join(data_path,"**","*.pdf"), recursive=True)
    # print(f"Found {len(pdf_files)} PDFs …")

    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

    try:
        import pdfplumber
    except ImportError:
        pdfplumber = None

    try:
        import pytesseract
        from PIL import Image as PILImage
        _ocr = True
    except ImportError:
        pytesseract = PILImage = None
        _ocr = False

    stats = dict(fitz=0, plumber=0, ocr=0, empty=0, errors=0)
    for fp in tqdm(pdf_files, desc="PDFs"):
        try:
            with open(fp,"rb") as fh:
                if fh.read(5) != b"%PDF-":
                    stats["errors"] += 1; continue
            text = ""
            if pymupdf and not text:
                try:
                    doc  = pymupdf.open(fp)
                    text = "\n".join(p.get_text() for p in doc).strip()
                    if text: stats["fitz"] += 1
                except: pass
            if pdfplumber and not text:
                try:
                    with pdfplumber.open(fp) as pdf:
                        text = "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
                    if text: stats["plumber"] += 1
                except: pass
            if not text and _ocr and pymupdf:
                try:
                    doc   = pymupdf.open(fp)
                    parts = []
                    for pn in range(min(3,len(doc))):
                        pix = doc[pn].get_pixmap(dpi=150, alpha=False)
                        img = PILImage.frombytes("RGB",[pix.width,pix.height],pix.samples)
                        parts.append(pytesseract.image_to_string(img))
                    text = "\n".join(parts).strip()
                    if text: stats["ocr"] += 1
                except: pass
            if not text:
                stats["empty"] += 1; continue
            r = weak_label(text)
            if r: examples.append(r)
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 3: print(f"  [err] {os.path.basename(fp)}: {e}")

    # print(f"PDFs — fitz:{stats['fitz']} plumber:{stats['plumber']} "
    #       f"ocr:{stats['ocr']} empty:{stats['empty']} err:{stats['errors']}")
    # print(f"Loaded {len(examples)} resume examples")
    return examples


# =============================================================================
# Train / Evaluate
# =============================================================================

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            out  = model(ids, mask, lbls)
            total_loss += out["loss"].item()
            for i, pred_seq in enumerate(out["predictions"]):
                lseq = lbls[i].cpu().numpy()
                tt, pp = [], []
                ptr = 0
                for lbl in lseq:
                    if lbl == -100:
                        if ptr < len(pred_seq): ptr += 1
                        continue
                    tt.append(ID2LABEL[int(lbl)])
                    pp.append(ID2LABEL[pred_seq[ptr]] if ptr < len(pred_seq) else "O")
                    ptr += 1
                if tt: all_labels.append(tt); all_preds.append(pp)
    return {"loss": total_loss/max(len(loader),1),
            "f1":   f1_score(all_labels, all_preds) if all_labels else 0.0,
            "labels": all_labels, "preds": all_preds}


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.fp16 and torch.cuda.is_available()

    print(f"\n{'='*60}\nHireSense AI — Local Training\nDevice: {device}  FP16: {use_amp}\n{'='*60}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    all_ex = load_resume_data(args.data_path) if (args.data_path and os.path.exists(args.data_path)) else []
    n_synth = max(300, 310 - len(all_ex))
    # print(f"Generating {n_synth} synthetic examples …")
    all_ex += [(t,l) for t,l in [ex[:2] if hasattr(ex,'tokens') else ex
                                  for ex in generate_synthetic(n_synth)]]

    train_ex, temp    = train_test_split(all_ex, test_size=0.2,  random_state=42)
    val_ex,   test_ex = train_test_split(temp,   test_size=0.5,  random_state=42)
    print(f"Train:{len(train_ex)}  Val:{len(val_ex)}  Test:{len(test_ex)}")

    nw = 0 if platform.system()=="Windows" else 4
    def mkloader(ex, bs, sh):
        ds = ResumeNERDataset(ex, tokenizer, args.max_length)
        return DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=nw, pin_memory=True)
    trl = mkloader(train_ex, args.batch_size,   True)
    vl  = mkloader(val_ex,   args.batch_size*2, False)
    tsl = mkloader(test_ex,  args.batch_size*2, False)

    model = BertBiLSTMCRF("bert-base-uncased", args.lstm_hidden, 2, 0.3, args.freeze_bert).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    bp  = [p for n,p in model.named_parameters() if "bert" in n and p.requires_grad]
    op  = [p for n,p in model.named_parameters() if "bert" not in n and p.requires_grad]
    opt = AdamW([{"params":bp,"lr":args.bert_lr},{"params":op,"lr":args.lstm_lr}], weight_decay=0.01)
    steps = len(trl)*args.epochs
    sch   = get_linear_schedule_with_warmup(opt, int(steps*0.1), steps)
    scaler = GradScaler() if use_amp else None

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = 0.0; patience = 0

    for epoch in range(1, args.epochs+1):
        model.train(); total_loss = 0.0; opt.zero_grad()
        bar = tqdm(trl, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(bar):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            if use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    out  = model(ids, mask, lbls); loss = out["loss"]/args.accumulation_steps
                scaler.scale(loss).backward()
                if (step+1)%args.accumulation_steps==0 or (step+1)==len(trl):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    prev=scaler.get_scale(); scaler.step(opt); scaler.update()
                    if scaler.get_scale()>=prev: sch.step()
                    opt.zero_grad()
            else:
                out  = model(ids, mask, lbls); loss = out["loss"]/args.accumulation_steps
                loss.backward()
                if (step+1)%args.accumulation_steps==0 or (step+1)==len(trl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    opt.step(); sch.step(); opt.zero_grad()
            total_loss += loss.item()*args.accumulation_steps
            bar.set_postfix({"loss":f"{loss.item()*args.accumulation_steps:.4f}"})

        vres = evaluate(model, vl, device)
        print(f"Epoch {epoch}: loss={total_loss/len(trl):.4f}  val_f1={vres['f1']:.4f}")

        if vres["f1"] > best_f1:
            best_f1 = float(vres["f1"]); patience = 0
            torch.save({"model_state_dict":model.state_dict(),"label2id":LABEL2ID,
                        "id2label":ID2LABEL,"f1":best_f1},
                       os.path.join(args.output_dir,"best_model.pt"))
            print(f"  ✓ Saved (F1={best_f1:.4f})")
        else:
            patience += 1
            if patience >= args.patience: print("Early stopping."); break

    ckpt = torch.load(os.path.join(args.output_dir,"best_model.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    tres = evaluate(model, tsl, device)
    print(f"\nTest F1: {tres['f1']:.4f}  Loss: {tres['loss']:.4f}")
    print(classification_report(tres["labels"], tres["preds"]))

    torch.save({"model_state_dict":model.state_dict(),"label2id":LABEL2ID,
                "id2label":ID2LABEL,"f1":float(tres["f1"])},
               os.path.join(args.output_dir,"resume_ner_model.pt"))
    tokenizer.save_pretrained(os.path.join(args.output_dir,"tokenizer"))
    with open(os.path.join(args.output_dir,"model_config.json"),"w") as f:
        json.dump({"bert_model_name":"bert-base-uncased","lstm_hidden_size":args.lstm_hidden,
                   "lstm_num_layers":2,"num_labels":NUM_LABELS,
                   "label2id":LABEL2ID,"id2label":{str(k):v for k,v in ID2LABEL.items()}},f,indent=2)
    print(f"Best Val F1: {best_f1:.4f}  |  Test F1: {tres['f1']:.4f}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",         type=str,   default="./data/kaggle_resume_pdf")
    p.add_argument("--output_dir",         type=str,   default="./output")
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--accumulation_steps", type=int,   default=4)
    p.add_argument("--epochs",             type=int,   default=6)
    p.add_argument("--max_length",         type=int,   default=512)
    p.add_argument("--lstm_hidden",        type=int,   default=256)
    p.add_argument("--bert_lr",            type=float, default=2e-5)
    p.add_argument("--lstm_lr",            type=float, default=1e-3)
    p.add_argument("--patience",           type=int,   default=5)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--fp16",               action="store_true")
    p.add_argument("--freeze_bert",        action="store_true")
    main(p.parse_args())

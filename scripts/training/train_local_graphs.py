"""
HireSense AI — Self-Contained Local Training Script
BERT + BiLSTM + CRF, Multi-Sector Resume NER

Usage:
    python train_local.py --data_path ./data/kaggle_resume_pdf --fp16

All logic is in one file (no imports from config/dataset/model).

Graphs saved to <output_dir>/plots/ after training completes:
  1. loss_curve.png            — Train vs Val loss per epoch
  2. f1_curve.png              — Val F1 per epoch + best-epoch marker
  3. lr_schedule.png           — Warmup + linear decay over every step
  4. label_distribution.png    — Entity span counts in training data
  5. entity_f1_bar.png         — Per-entity F1 on test set
  6. entity_pr_scatter.png     — Precision vs Recall bubble chart
  7. confusion_matrix.png      — What entity types get confused with what
  8. example_annotation.png    — Coloured token-level prediction on a CV
"""

import os, re, json, glob, random, argparse, platform
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

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
LABEL2ID   = {l: i for i, l in enumerate(ENTITY_LABELS)}
ID2LABEL   = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(ENTITY_LABELS)

ENTITY_TYPES = [
    "SKILL","EXP","EDU","PROJ","ACH","CERT",
    "ORG","LOC","DATE","NAME","CONTACT",
]

RESUME_SECTION_HEADERS = {
    "education","experience","skills","projects","achievements",
    "certifications","awards","summary","objective","profile","contact",
    "internships","positions","extracurricular","activities","interests",
    "startup","competitions","academic",
}

# One colour per entity type — used consistently across all plots
ENTITY_COLORS = {
    "SKILL":   "#4CAF50", "EXP":     "#2196F3", "EDU":     "#9C27B0",
    "PROJ":    "#FF9800", "ACH":     "#F44336", "CERT":    "#00BCD4",
    "ORG":     "#795548", "LOC":     "#607D8B", "DATE":    "#FFC107",
    "NAME":    "#E91E63", "CONTACT": "#009688", "SECTOR":  "#3F51B5",
    "O":       "#EEEEEE",
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
                            dropout=lstm_dropout if lstm_layers > 1 else 0)
        self.dropout    = nn.Dropout(lstm_dropout)
        self.hidden2tag = nn.Linear(lstm_hidden * 2, NUM_LABELS)
        self.crf        = CRF(NUM_LABELS, batch_first=True)
        self._init()

    def _init(self):
        for n, p in self.lstm.named_parameters():
            if "weight_ih" in n:   nn.init.xavier_uniform_(p)
            elif "weight_hh" in n: nn.init.orthogonal_(p)
            elif "bias" in n:
                nn.init.zeros_(p)
                sz = p.size(0); p.data[sz // 4 : sz // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)

    def forward(self, ids, mask, labels=None):
        seq   = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        seq,_ = self.lstm(seq)
        seq   = self.dropout(seq)
        emit  = self.hidden2tag(seq).float()
        m     = mask.bool()
        out   = {"emissions": emit}
        if labels is not None:
            lc = labels.clone(); lc[labels == -100] = 0
            out["loss"] = -self.crf(emit, lc, mask=m, reduction="mean")
        with torch.no_grad():
            out["predictions"] = self.crf.decode(emit, mask=m)
        return out


# =============================================================================
# Weak labeller
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
DEGREE_SINGLE = {
    "bachelor","master","phd","doctorate","mba","btech","mtech","b.tech","m.tech",
    "b.sc","m.sc","b.com","m.com","b.e","m.e","b.a","m.a","llb","llm","mbbs",
    "md","pgdm","ca","cfa","cpa","diploma","b.s","m.s",
    "university","college","institute","iit","nit","bits","iiit",
}
CERT_SINGLE = {
    "certified","certification","certificate","oracle","bloomberg",
    "nptel","coursera","udemy","hackerrank","freecodecamp","scalar",
    "cs50","pmp","cissp","ceh","comptia","databricks",
}
PROJ_SINGLE  = {"project","projects","built","developed","implemented",
                "created","designed","deployed","architected","engineered"}
ACH_SINGLE   = {"award","winner","finalist","recognition","published","patent",
                "honors","scholarship","champion","medal","ranked","selected","hackathon"}
EXP_SINGLE   = {
    "senior","junior","lead","principal","associate","head",
    "engineer","developer","analyst","scientist","architect","designer",
    "manager","director","officer","specialist","intern","internship",
    "founder","co-founder","president","consultant","trainee","freelancer",
}
ORG_SINGLE = {
    "google","microsoft","amazon","meta","apple","tcs","infosys","wipro",
    "hcl","cognizant","capgemini","accenture","jio","reliance","flipkart",
    "deloitte","pwc","ey","kpmg","hdfc","icici","nysonaiconnect","nsoft",
    "hacktify","mactores","cognifyz","meryt","isf","gostops",
}
LOC_SINGLE = {
    "jabalpur","mumbai","pune","bangalore","bengaluru","delhi","noida",
    "hyderabad","chennai","kolkata","vellore","jaipur","surat","india","mp",
}

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
ROLE_WORDS = {
    "engineer","developer","analyst","scientist","architect","designer",
    "manager","director","officer","specialist","consultant","intern",
    "coordinator","executive","founder","co-founder","president","technician",
}
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


def _build_token_offsets(text, tokens):
    offsets, pos = [], 0
    for tok in tokens:
        idx = text.find(tok, pos)
        if idx == -1: offsets.append((-1, -1))
        else:         offsets.append((idx, idx + len(tok))); pos = idx + 1
    return offsets


def weak_label(text):
    if not text or len(text.strip()) < 20: return None
    tokens = re.findall(
        r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|[\w]+(?:[./+\-#][\w]+)*|\S", text)
    if len(tokens) < 5: return None
    labels = ["O"] * len(tokens)
    tl     = [t.lower() for t in tokens]

    def safe(i, base, start):
        if 0 <= i < len(labels) and labels[i] == "O":
            labels[i] = f"B-{base}" if start else f"I-{base}"

    tok_offsets = _build_token_offsets(text, tokens)

    for m in DATE_RE.finditer(text):
        first = True
        for ti, (s, _) in enumerate(tok_offsets):
            if s != -1 and m.start() <= s < m.end():
                safe(ti, "DATE", first); first = False

    for ti, tok in enumerate(tokens):
        if EMAIL_RE.fullmatch(tok):                   safe(ti, "CONTACT", True)
        elif PHONE_RE.fullmatch(tok.replace(" ", "")): safe(ti, "CONTACT", True)

    for m in ACH_PHRASE_RE.finditer(text):
        span = m.group(0).lower().split()
        for ti in range(len(tokens) - len(span) + 1):
            if [tl[ti+k] for k in range(len(span))] == span:
                for k in range(len(span)): safe(ti+k, "ACH", k == 0)

    i = 0
    while i < len(tokens):
        matched = False
        for phrase in PHRASE_SKILLS:
            end = i + len(phrase)
            if end <= len(tokens) and tuple(tl[i:end]) == phrase:
                for j in range(len(phrase)): safe(i+j, "SKILL", j == 0)
                i += len(phrase); matched = True; break
        if not matched: i += 1

    i = 0
    while i < len(tokens):
        if tl[i] in SENIORITY and i+1 < len(tokens) and tl[i+1] in ROLE_WORDS:
            safe(i, "EXP", True); safe(i+1, "EXP", False)
            j = i + 2
            while j < len(tokens) and j < i+4 and tl[j] in ROLE_WORDS:
                safe(j, "EXP", False); j += 1
            i = j
        else: i += 1

    for i, tok in enumerate(tl):
        if tok in RESUME_SECTION_HEADERS: continue
        if   tok in SKILL_SINGLE:  safe(i, "SKILL",   True)
        elif tok in DEGREE_SINGLE: safe(i, "EDU",     True)
        elif tok in CERT_SINGLE:   safe(i, "CERT",    True)
        elif tok in PROJ_SINGLE:   safe(i, "PROJ",    True)
        elif tok in ACH_SINGLE:    safe(i, "ACH",     True)
        elif tok in EXP_SINGLE:    safe(i, "EXP",     True)
        elif tok in ORG_SINGLE:    safe(i, "ORG",     True)
        elif tok in LOC_SINGLE:    safe(i, "LOC",     True)
        elif CGPA_RE.match(tokens[i]): safe(i, "EDU", True)

    return (tokens, labels)


# =============================================================================
# Synthetic data
# =============================================================================

def generate_synthetic(n=2000):
    SECTOR_DATA = [
        {
            "sector": "cs_developer",
            "skills": [
                ["Python","JavaScript","React","Node.js","MongoDB","Docker","AWS","JWT"],
                ["Java","C++","SQL","Docker","Git","TypeScript","PostgreSQL","Flask"],
                ["Python","FastAPI","MySQL","Redis","Linux","Docker","Firebase"],
            ],
            "titles": [
                "Full Stack Developer Intern","Software Engineer Intern",
                "Product Engineer Intern","App Development Intern",
            ],
            "orgs": ["Google","Microsoft","TCS","Infosys","NysonAiConnect","Jio Platforms"],
            "degrees": [
                "Bachelor of Technology in Information Technology",
                "Bachelor of Technology in Computer Science",
            ],
            "colleges": [
                "Jabalpur Engineering College",
                "Vellore Institute of Technology",
                "NIT Trichy",
            ],
            "ach": [
                "Solved 500+ DSA problems LeetCode GeeksForGeeks",
                "Finalist HackCrux 2025 LNMIIT Jaipur",
                "1st Place HackCrux 2025 90+ teams",
                "Smart India Hackathon 2024 Semifinalist",
            ],
            "certs": [
                "Oracle Cloud Infrastructure Architect Associate",
                "Full-Stack Web Development MERN CodeHelp",
                "Machine Learning with Python freecodecamp",
            ],
        },
        {
            "sector": "data_science",
            "skills": [
                ["Python","PyTorch","TensorFlow","scikit-learn","Pandas","NumPy","BERT"],
                ["Python","LangChain","Gemini","RAG","FastAPI","Pinecone","Docker","GCP"],
                ["Python","XGBoost","SHAP","Flask","Docker","GCP","CNN","LSTM"],
            ],
            "titles": [
                "Data Science Intern","Machine Learning Engineer Intern",
                "Founder Lead Engineer","Data Analyst Intern",
            ],
            "orgs": ["Cognifyz Technologies","Mactores","GoStops","MERYT","Jio Platforms"],
            "degrees": [
                "B.S. Degree Data Science Applications",
                "Bachelor of Technology Computer Science",
            ],
            "colleges": ["IIT Madras","Jabalpur Engineering College","VIT Vellore"],
            "ach": [
                "Top Finalist 10000+ teams Google Cloud Agentic AI Day 2025",
                "Finalist Top 91/700+ Google Cloud Agentic AI Day 2025 Bangalore",
                "Reliance Foundation Undergraduate Scholar",
            ],
            "certs": [
                "Oracle Cloud Infrastructure 2025 Certified Data Science Professional",
                "AWS Certified Cloud Practitioner",
                "Databricks Certified Data Engineer Associate",
            ],
        },
        {
            "sector": "cybersecurity",
            "skills": [
                ["Python","Kali Linux","Wireshark","Metasploit","Nmap","Splunk"],
                ["Python","C","Linux","XSS","SQL Injection","IDOR","SIEM"],
            ],
            "titles": [
                "CyberSecurity Intern","Linux Trainee","Security Analyst Intern",
            ],
            "orgs": ["Hacktify Cyber Security","CYBERSEC INFOTECH","IBM"],
            "degrees": [
                "Bachelor of Technology Artificial Intelligence Data Science",
                "Bachelor of Technology Information Technology",
            ],
            "colleges": ["Jabalpur Engineering College","VIT Vellore"],
            "ach": [
                "CTF challenges red teaming OSINT cryptography",
                "Smart India Hackathon 2024 Semifinals 500+ teams",
            ],
            "certs": [
                "Google Cybersecurity Professional Certification",
                "Cyber Security Privacy NPTEL",
                "Ethical Hacking NPTEL",
            ],
        },
        {
            "sector": "finance_tech",
            "skills": [
                ["Python","SQL","Microsoft Excel","Tableau","Power BI","Financial Modelling"],
                ["Bloomberg","MySQL","NumPy","Pandas","Risk Management","FastAPI"],
            ],
            "titles": [
                "Technical Intern","Finance Intern","Data Analyst Intern",
                "Junior Consultant Intern",
            ],
            "orgs": [
                "Jio Platforms Ltd","Ernst Young EY","Deloitte",
                "PwC","J.P. Morgan","HOSHŌ DIGITAL",
            ],
            "degrees": ["MBA Tech","BTech Computer","Bachelor of Management Studies"],
            "colleges": [
                "MPSTME NMIMS Mumbai",
                "Shaheed Sukhdev College Business Studies University Delhi",
            ],
            "ach": [
                "Bloomberg Market Concepts Certificate",
                "ADAPT 3.0 Winner Algorithmic Trading",
                "Merit Certificate outstanding internship performance",
            ],
            "certs": [
                "Bloomberg Finance Fundamentals",
                "Bloomberg Market Concepts",
                "Infosys Financial Modeling",
            ],
        },
    ]

    FIRST = [
        "Priya","Rahul","Ananya","Arjun","Sneha","Divya","Karan","Neha",
        "Vikram","Rohit","Aditya","Sanidhya","Monal","Sonu","Jayendra",
        "Siddharth","Anshika","Arya","Shravan","Mansi","Ishan","Sanyam",
        "Kinjal","Vansh","Subrato","Sanskar","Samkit","Shreeansh",
    ]
    LAST = [
        "Sharma","Patel","Gupta","Singh","Jain","Tiwari","Agarwal",
        "Kumar","Narang","Sanghvi","Israni","Vishwakarma","Bhagat","Saxena","Tomar",
    ]

    examples = []
    for _ in range(n):
        d = random.choice(SECTOR_DATA)
        t, l = [], []

        def add(words, tag):
            for j, w in enumerate(words):
                t.append(w); l.append(f"B-{tag}" if j == 0 else f"I-{tag}")

        def o(w):
            t.append(w); l.append("O")

        add(f"{random.choice(FIRST)} {random.choice(LAST)}".split(), "NAME")

        if random.random() > 0.3:
            o("\nEmail:")
            o(f"{random.choice(FIRST).lower()}{random.randint(10,999)}@gmail.com")
        if random.random() > 0.5:
            o("Phone:")
            t.append(f"+91{random.randint(7000000000,9999999999)}"); l.append("B-CONTACT")

        sections = ["skills","experience","education","projects","achievements","certs"]
        random.shuffle(sections)

        for sec in sections:
            if sec == "skills":
                o("\n" + random.choice(["SKILLS","Skills","Technical Skills"]))
                sset = random.choice(d["skills"])
                for sk in random.sample(sset, min(random.randint(3,6), len(sset))):
                    if random.random() > 0.5: o("•")
                    add(sk.split(), "SKILL")

            elif sec == "experience":
                o("\n" + random.choice(["Experience","WORK EXPERIENCE","Internships"]))
                for _ in range(random.randint(1,3)):
                    role = random.choice(d["titles"]); org = random.choice(d["orgs"])
                    if random.random() > 0.5:
                        add(role.split(), "EXP"); o("at"); add(org.split(), "ORG")
                    else:
                        add(org.split(), "ORG"); o("-"); add(role.split(), "EXP")
                    yr = random.randint(2020, 2024)
                    o("("); t.append(str(yr)); l.append("B-DATE")
                    o("-"); t.append(random.choice(["Present", str(yr+1)])); l.append("B-DATE")
                    o(")")
                    o(random.choice(["Worked on scalable systems","Built dashboards",
                                     "Collaborated with teams","Improved model performance"]))

            elif sec == "education":
                o("\n" + random.choice(["Education","ACADEMIC DETAILS"]))
                for _ in range(random.randint(1,2)):
                    add(random.choice(d["degrees"]).split(), "EDU")
                    o(","); add(random.choice(d["colleges"]).split(), "ORG")
                    t.append(str(round(random.uniform(6.0, 9.8), 2))); l.append("B-EDU")

            elif sec == "projects":
                o("\nPROJECTS")
                for _ in range(random.randint(1,3)):
                    o("•"); o(random.choice([
                        "AI Chatbot using NLP","Stock Prediction using LSTM",
                        "Portfolio Optimization System","Fraud Detection Model",
                        "Resume Builder App","Job Preparation Platform",
                    ]))

            elif sec == "achievements":
                o("\nAchievements")
                add(random.choice(d["ach"]).split(), "ACH")

            elif sec == "certs":
                o("\nCertifications")
                add(random.choice(d["certs"]).split(), "CERT")

        if random.random() > 0.5:
            o("|"); t.append(random.choice(["Jabalpur","Mumbai","Pune","Bangalore","Delhi"]))
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
        enc = self.tokenizer(
            tokens, is_split_into_words=True,
            max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        wids = enc.word_ids(); aligned = []; prev = None
        for wid in wids:
            if wid is None: aligned.append(-100)
            elif wid != prev:
                raw = labels[wid] if wid < len(labels) else "O"
                aligned.append(LABEL2ID.get(raw, 0))
            else:
                raw = labels[wid] if wid < len(labels) else "O"
                if raw.startswith("B-"): raw = "I-" + raw[2:]
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
    for fp in glob.glob(os.path.join(data_path, "**", "*.txt"), recursive=True):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f: text = f.read()
            r = weak_label(text)
            if r: examples.append(r)
        except: pass

    try:
        import pandas as pd
        for fp in glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True):
            try:
                df  = pd.read_csv(fp)
                col = next((c for c in df.columns if c.lower() in
                            {"resume","resume_str","text","cv","content"}), None)
                if col:
                    for txt in df[col].dropna().astype(str):
                        r = weak_label(txt)
                        if r: examples.append(r)
            except: pass
    except ImportError: pass

    pdf_files = glob.glob(os.path.join(data_path, "**", "*.pdf"), recursive=True)
    try:
        import fitz as pymupdf
    except ImportError: pymupdf = None
    try:
        import pdfplumber
    except ImportError: pdfplumber = None
    try:
        import pytesseract
        from PIL import Image as PILImage; _ocr = True
    except ImportError: pytesseract = PILImage = None; _ocr = False

    stats = dict(fitz=0, plumber=0, ocr=0, empty=0, errors=0)
    for fp in tqdm(pdf_files, desc="PDFs"):
        try:
            with open(fp, "rb") as fh:
                if fh.read(5) != b"%PDF-": stats["errors"] += 1; continue
            text = ""
            if pymupdf and not text:
                try:
                    doc = pymupdf.open(fp)
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
                    doc = pymupdf.open(fp); parts = []
                    for pn in range(min(3, len(doc))):
                        pix = doc[pn].get_pixmap(dpi=150, alpha=False)
                        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        parts.append(pytesseract.image_to_string(img))
                    text = "\n".join(parts).strip()
                    if text: stats["ocr"] += 1
                except: pass
            if not text: stats["empty"] += 1; continue
            r = weak_label(text)
            if r: examples.append(r)
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 3: print(f"  [err] {os.path.basename(fp)}: {e}")
    return examples


# =============================================================================
# Evaluate
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
                lseq = lbls[i].cpu().numpy(); tt, pp = [], []; ptr = 0
                for lbl in lseq:
                    if lbl == -100:
                        if ptr < len(pred_seq): ptr += 1
                        continue
                    tt.append(ID2LABEL[int(lbl)])
                    pp.append(ID2LABEL[pred_seq[ptr]] if ptr < len(pred_seq) else "O")
                    ptr += 1
                if tt: all_labels.append(tt); all_preds.append(pp)
    return {"loss":   total_loss / max(len(loader), 1),
            "f1":     f1_score(all_labels, all_preds) if all_labels else 0.0,
            "labels": all_labels, "preds": all_preds}


# =============================================================================
# ██████╗ ██╗      ██████╗ ████████╗███████╗
# ██╔══██╗██║     ██╔═══██╗╚══██╔══╝██╔════╝
# ██████╔╝██║     ██║   ██║   ██║   ███████╗
# ██╔═══╝ ██║     ██║   ██║   ██║   ╚════██║
# ██║     ███████╗╚██████╔╝   ██║   ███████║
# ╚═╝     ╚══════╝ ╚═════╝    ╚═╝   ╚══════╝
# =============================================================================

def _mpl():
    """Return (plt, mpatches) with clean style applied."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams.update({
        "figure.dpi":         150,
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#E0E0E0",
        "grid.linewidth":     0.8,
        "legend.framealpha":  0.92,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })
    return plt, mpatches


# ─────────────────────────────────────────────────────────────────────────────
# 1. Train vs Val loss
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(history, output_dir):
    """
    Train loss vs Val loss per epoch.
    - Both curves falling = model is learning normally.
    - Val > Train by a large margin = overfitting (model memorising training data).
    - Both curves flat and high = underfitting (model hasn't learned enough).
    The shaded area between the two lines visualises the overfitting gap.
    """
    plt, _ = _mpl()
    epochs = [h["epoch"] for h in history]
    tl = [h["train_loss"] for h in history]
    vl = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, tl, "o-",  color="#2196F3", lw=2, ms=6, label="Train Loss")
    ax.plot(epochs, vl, "s--", color="#F44336", lw=2, ms=6, label="Val Loss")
    ax.fill_between(epochs, tl, vl, alpha=0.09, color="#9C27B0",
                    label="Overfitting gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CRF Negative Log-Likelihood Loss")
    ax.set_title("Training vs Validation Loss\n"
                 "(Ideal: both curves fall together without diverging)")
    ax.set_xticks(epochs)
    ax.legend(); ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "loss_curve.png")
    plt.savefig(path); plt.close(); print(f"  📊 {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Validation F1 over epochs
# ─────────────────────────────────────────────────────────────────────────────

def plot_f1_curve(history, best_epoch, output_dir):
    """
    How NER quality (F1) improves each epoch.
    F1 = harmonic mean of Precision and Recall, measured at entity-span level.
    The orange star marks the best checkpoint that was saved.
    The red shaded region = patience window (early stopping fired here).
    """
    plt, _ = _mpl()
    epochs = [h["epoch"]  for h in history]
    f1s    = [h["val_f1"] for h in history]
    best   = max(f1s)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, f1s, "D-", color="#4CAF50", lw=2, ms=7, label="Val F1")
    ax.axhline(best, color="#FF9800", ls="--", lw=1.5,
               label=f"Best F1 = {best:.4f}")
    bidx = f1s.index(best)
    ax.plot(epochs[bidx], best, "*", color="#FF5722", ms=18, zorder=5,
            label=f"Best epoch ({epochs[bidx]})")
    if len(epochs) > bidx + 1:
        ax.axvspan(epochs[bidx], epochs[-1], alpha=0.06, color="#F44336",
                   label="Early-stopping patience")
    ax.set_xlabel("Epoch"); ax.set_ylabel("seqeval F1 (entity-level)")
    ax.set_title("Validation F1 Score per Epoch\n"
                 "(Higher = model finds more correct entity spans)")
    ax.set_xticks(epochs)
    ax.set_ylim(0, min(1.08, best * 1.35 + 0.06))
    ax.legend(fontsize=9); ax.grid(True, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "f1_curve.png")
    plt.savefig(path); plt.close(); print(f"  📊 {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Learning rate schedule
# ─────────────────────────────────────────────────────────────────────────────

def plot_lr_schedule(total_steps, warmup_steps, bert_lr, lstm_lr, output_dir):
    """
    Shows LR at every training step for both parameter groups.
    Warmup phase: LR rises linearly 0 → peak.
      Why? Starting BERT fine-tuning at full LR would destroy the pre-trained
      weights. A slow warmup lets the model adapt gradually.
    Decay phase: LR falls linearly → 0.
      Why? Smaller updates in later epochs prevent overshooting the optimum.
    BERT uses a 50× smaller LR than BiLSTM/CRF because it's already
    pre-trained and needs gentle nudging, not big changes.
    """
    plt, _ = _mpl()
    steps = np.arange(total_steps)

    def lr_sched(step, peak):
        if step < warmup_steps:
            return peak * step / max(warmup_steps, 1)
        frac = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return peak * max(0.0, 1.0 - frac)

    bert_lrs = [lr_sched(s, bert_lr) for s in steps]
    lstm_lrs = [lr_sched(s, lstm_lr) for s in steps]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, bert_lrs, color="#2196F3", lw=2,
            label=f"BERT LR  (peak = {bert_lr:.0e})")
    ax.plot(steps, lstm_lrs, color="#FF9800", lw=2,
            label=f"BiLSTM/CRF LR  (peak = {lstm_lr:.0e})")
    ax.axvline(warmup_steps, color="#9E9E9E", ls=":", lw=1.5,
               label=f"Warmup ends at step {warmup_steps}")
    ax.axvspan(0, warmup_steps, alpha=0.06, color="#4CAF50", label="Warmup phase")
    ax.axvspan(warmup_steps, total_steps, alpha=0.04, color="#F44336", label="Decay phase")
    ax.set_xlabel("Training Step"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule\n"
                 "BERT gets 50× smaller LR than BiLSTM/CRF to protect pre-trained weights")
    ax.legend(fontsize=9); ax.grid(True, axis="y")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "lr_schedule.png")
    plt.savefig(path); plt.close(); print(f"  📊 {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Label distribution in training data
# ─────────────────────────────────────────────────────────────────────────────

def plot_label_distribution(train_examples, output_dir):
    """
    How many labelled entity spans of each type exist in the training set.
    This directly explains F1 differences:
      A bar that's 10× taller than another = model saw 10× more examples.
      Rare entity types almost always score lower — more data fixes this.
    """
    plt, _ = _mpl()
    counts = defaultdict(int)
    for tokens, labels in train_examples:
        for lbl in labels:
            if lbl.startswith("B-"): counts[lbl[2:]] += 1

    if not counts: return
    types  = sorted(counts, key=counts.get, reverse=True)
    values = [counts[t] for t in types]
    colors = [ENTITY_COLORS.get(t, "#90A4AE") for t in types]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(types, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values)*0.01,
                f"{val:,}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xlabel("Entity Type"); ax.set_ylabel("Labelled Spans in Training Data")
    ax.set_title("Training Data Entity Distribution\n"
                 "(Taller bar = model saw more examples → usually higher F1)")
    ax.grid(True, axis="y"); plt.tight_layout()
    path = os.path.join(output_dir, "plots", "label_distribution.png")
    plt.savefig(path); plt.close(); print(f"  📊 {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-entity F1 bar chart
# ─────────────────────────────────────────────────────────────────────────────

def _per_entity_metrics(all_true, all_pred):
    report = classification_report(all_true, all_pred,
                                   output_dict=True, zero_division=0)
    out = {}
    for key, vals in report.items():
        if key in ("micro avg","macro avg","weighted avg","accuracy"): continue
        out[key] = {"precision": vals.get("precision",0.0),
                    "recall":    vals.get("recall",   0.0),
                    "f1":        vals.get("f1-score", 0.0),
                    "support":   int(vals.get("support",0))}
    return out


def plot_entity_f1_bar(all_true, all_pred, output_dir):
    """
    F1 score broken down per entity type on the test set.
    Sorted best → worst so weaknesses are immediately obvious.
    The number inside each bar is how many test spans existed for that type.
    Low score + large n = the model genuinely struggles here.
    Low score + small n = just not enough test examples to be reliable.
    """
    plt, _ = _mpl()
    m = _per_entity_metrics(all_true, all_pred)
    if not m: return

    types  = sorted(m, key=lambda t: m[t]["f1"], reverse=True)
    f1s    = [m[t]["f1"]     for t in types]
    sups   = [m[t]["support"] for t in types]
    colors = [ENTITY_COLORS.get(t, "#90A4AE") for t in types]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(types, f1s, color=colors, edgecolor="white", linewidth=0.8)
    for bar, f1, s in zip(bars, f1s, sups):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{f1:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if bar.get_height() > 0.12:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                    f"n={s}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.axhline(0.70, color="#9E9E9E", ls="--", lw=1.2, label="0.70 reference")
    ax.set_ylim(0, 1.12); ax.set_xlabel("Entity Type")
    ax.set_ylabel("F1 Score (test set)")
    ax.set_title("Per-Entity-Type F1 on Test Set\n"
                 "(Number inside bar = test spans for that type)")
    ax.legend(fontsize=9); ax.grid(True, axis="y"); plt.tight_layout()
    path = os.path.join(output_dir, "plots", "entity_f1_bar.png")
    plt.savefig(path); plt.close(); print(f"  📊 {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Precision vs Recall scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_scatter(all_true, all_pred, output_dir):
    """
    Each dot = one entity type. Bubble size = number of test examples.
    What the quadrants mean:
      Top-right: high Precision AND high Recall = ideal.
      Top-left:  high Precision, low Recall = model is very selective —
                 it only predicts when confident, misses many real entities.
      Bottom-right: low Precision, high Recall = model over-predicts —
                    finds most real entities but also many false ones.
      Bottom-left: both low = model is struggling with this type.
    Dashed iso-lines show equal F1 contours (0.5, 0.7, 0.9).
    """
    plt, _ = _mpl()
    m = _per_entity_metrics(all_true, all_pred)
    if not m: return

    fig, ax = plt.subplots(figsize=(8, 7))
    for etype, vals in m.items():
        p   = vals["precision"]; r = vals["recall"]
        s   = max(vals["support"], 1)
        col = ENTITY_COLORS.get(etype, "#90A4AE")
        ax.scatter(r, p, s=s*4, c=col, alpha=0.85,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(etype, (r, p), textcoords="offset points",
                    xytext=(7, 5), fontsize=9, fontweight="bold", color="#333333")

    ax.axhline(0.7, color="#BDBDBD", ls="--", lw=1)
    ax.axvline(0.7, color="#BDBDBD", ls="--", lw=1)
    ax.text(0.02, 0.72, "P=0.70", fontsize=8, color="#9E9E9E")
    ax.text(0.72, 0.02, "R=0.70", fontsize=8, color="#9E9E9E")

    for f1_iso in [0.5, 0.7, 0.9]:
        pts = np.linspace(0.01, 1.0, 200)
        p_l = f1_iso * pts / (2*pts - f1_iso + 1e-9)
        mask = (p_l >= 0) & (p_l <= 1)
        ax.plot(pts[mask], p_l[mask], ":", color="#CFD8DC", lw=1)
        ax.text(pts[mask][-1]-0.06, p_l[mask][-1],
                f"F1={f1_iso}", fontsize=7.5, color="#90A4AE")

    ax.set_xlim(-0.05, 1.12); ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Recall  —  did the model find all real entities?")
    ax.set_ylabel("Precision  —  when it predicts, is it right?")
    ax.set_title("Precision vs Recall per Entity Type\n"
                 "(bubble size = number of test examples; top-right = ideal)")
    ax.grid(True, alpha=0.4); plt.tight_layout()
    path = os.path.join(output_dir, "plots", "entity_pr_scatter.png")
    plt.savefig(path); plt.close(); print(f"  📊 {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Entity-type confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(all_true, all_pred, output_dir):
    """
    Rows = what the entity ACTUALLY is. Columns = what the model PREDICTED.
    Diagonal cells (blue) = correct predictions.
    Off-diagonal cells = confusions. For example:
      Row=EXP, Col=ORG means a real EXP token was wrongly tagged as ORG.
    Large off-diagonal numbers identify specific weaknesses to fix next.
    Only tokens that truly belong to an entity type are shown in the rows
    (O-tag rows are excluded to keep the chart readable).
    """
    plt, _ = _mpl()
    true_flat = [lbl for seq in all_true for lbl in seq]
    pred_flat = [lbl for seq in all_pred for lbl in seq]

    def etype(tag):
        if tag == "O": return None
        return tag[2:] if "-" in tag else tag

    types = sorted(set(t for t in (etype(l) for l in true_flat) if t))
    if not types: return

    n = len(types); t2i = {t: i for i, t in enumerate(types)}
    mat = np.zeros((n, n), dtype=int)
    for tt, pt in zip(true_flat, pred_flat):
        et = etype(tt); ep = etype(pt)
        if et is None: continue
        pi = t2i.get(ep, -1)
        if pi != -1: mat[t2i[et]][pi] += 1

    fig, ax = plt.subplots(figsize=(max(8, n*0.9), max(6, n*0.8)))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(types, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(types)
    ax.set_xlabel("Predicted Type"); ax.set_ylabel("True Type")
    ax.set_title("Entity-Type Confusion Matrix (test set)\n"
                 "Diagonal = correct  |  Off-diagonal = what gets confused with what")

    vmax = mat.max() or 1
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if v == 0: continue
            col = "white" if v > vmax * 0.5 else "#212121"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=8, color=col, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "confusion_matrix.png")
    plt.savefig(path); plt.close(); print(f"  📊 {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Coloured token annotation on a sample CV
# ─────────────────────────────────────────────────────────────────────────────

def plot_example_annotation(model, tokenizer, device, output_dir, max_tokens=80):
    """
    Runs the trained model on a realistic CV snippet and draws a grid
    where each word is coloured by its predicted entity type.
    ▶ = start of a new entity span (B- tag).
    ⋯ = continuation of the previous span (I- tag).
      (plain) = not an entity (O tag).
    This is the most intuitive way to see what the model actually does.
    """
    plt, mpatches = _mpl()

    sample_cv = (
        "Priya Sharma | priya.sharma@gmail.com | +919876543210 | Mumbai\n"
        "Education: Bachelor of Technology in Computer Science, "
        "Jabalpur Engineering College, CGPA 8.71\n"
        "Experience: Senior Software Engineer at Google (2021-Present)\n"
        "Experience: Data Science Intern at Cognifyz Technologies (2020-2021)\n"
        "Skills: Python PyTorch FastAPI Docker AWS React PostgreSQL Git\n"
        "Projects: Built an AI resume screener using BERT and FastAPI "
        "deployed on GCP with Docker\n"
        "Achievements: Finalist HackCrux 2025 LNMIIT Jaipur "
        "Solved 500+ DSA problems LeetCode\n"
        "Certifications: Oracle Cloud Infrastructure Architect Associate "
        "AWS Certified Developer"
    )

    tokens = re.findall(
        r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|[\w]+(?:[./+\-#][\w]+)*|\S",
        sample_cv,
    )[:max_tokens]

    enc = tokenizer(tokens, is_split_into_words=True,
                    max_length=512, padding="max_length",
                    truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        out   = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        preds = out["predictions"][0]

    word_ids   = enc.word_ids()
    word_preds = {}
    ptr        = 0
    for wid in word_ids:
        if wid is None: ptr += 1; continue
        if ptr >= len(preds): break
        if wid not in word_preds:
            word_preds[wid] = ID2LABEL.get(preds[ptr], "O")
        ptr += 1

    tok_labels = [word_preds.get(i, "O") for i in range(len(tokens))]

    COLS = 10
    rows = (len(tokens) + COLS - 1) // COLS
    fig, ax = plt.subplots(figsize=(16, max(4, rows * 1.1 + 2.8)))
    ax.set_xlim(0, COLS); ax.set_ylim(0, rows + 0.5); ax.axis("off")
    ax.set_title(
        "Token-Level NER Prediction on a Sample CV\n"
        "▶ = entity start  ⋯ = continuation  (plain) = not an entity",
        fontsize=13, fontweight="bold", pad=12,
    )

    for idx, (tok, lbl) in enumerate(zip(tokens, tok_labels)):
        row = rows - 1 - (idx // COLS)
        col = idx % COLS
        etype  = lbl[2:] if lbl.startswith(("B-", "I-")) else "O"
        color  = ENTITY_COLORS.get(etype, "#EEEEEE")
        prefix = ("▶ " if lbl.startswith("B-") else
                  "⋯ " if lbl.startswith("I-") else "  ")

        rect = plt.Rectangle(
            (col + 0.04, row + 0.10), 0.88, 0.74,
            facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.88,
        )
        ax.add_patch(rect)

        display  = (tok[:8] + "…") if len(tok) > 9 else tok
        txt_col  = ("white" if etype not in ("O", "DATE", "CONTACT")
                    else "#212121")
        ax.text(col + 0.50, row + 0.53, prefix + display,
                ha="center", va="center", fontsize=7.8,
                color=txt_col, fontweight="bold" if etype != "O" else "normal")
        if etype != "O":
            ax.text(col + 0.50, row + 0.19, etype,
                    ha="center", va="center", fontsize=6.5,
                    color=txt_col, style="italic")

    legend_handles = [
        mpatches.Patch(facecolor=ENTITY_COLORS[et], edgecolor="white",
                       label=et, linewidth=0.8)
        for et in ENTITY_TYPES
    ] + [mpatches.Patch(facecolor="#EEEEEE", edgecolor="#BDBDBD",
                        label="O  (not an entity)", linewidth=0.8)]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.07),
              ncol=min(6, len(legend_handles)),
              fontsize=8.5, framealpha=0.95)
    plt.tight_layout()
    path = os.path.join(output_dir, "plots", "example_annotation.png")
    plt.savefig(path); plt.close(); print(f"  📊 {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Master plot function — called once after training
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(history, best_epoch, total_steps, warmup_steps,
                        bert_lr, lstm_lr, train_examples, test_true, test_pred,
                        model, tokenizer, device, output_dir):
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    print("\n" + "=" * 55)
    print("  Generating visualisation plots …")
    print("=" * 55)

    plot_loss_curve(history, output_dir)
    plot_f1_curve(history, best_epoch, output_dir)
    plot_lr_schedule(total_steps, warmup_steps, bert_lr, lstm_lr, output_dir)
    plot_label_distribution(train_examples, output_dir)
    plot_entity_f1_bar(test_true, test_pred, output_dir)
    plot_pr_scatter(test_true, test_pred, output_dir)
    plot_confusion_matrix(test_true, test_pred, output_dir)
    plot_example_annotation(model, tokenizer, device, output_dir)

    plots_dir = os.path.join(output_dir, "plots")
    print(f"\n  All 8 plots saved → {plots_dir}/")
    print("  ─────────────────────────────────────────────────")
    print("  1  loss_curve.png            Train vs Val loss")
    print("  2  f1_curve.png              Validation F1 + best epoch")
    print("  3  lr_schedule.png           Warmup + linear LR decay")
    print("  4  label_distribution.png    Entity counts in training data")
    print("  5  entity_f1_bar.png         Per-entity F1 on test set")
    print("  6  entity_pr_scatter.png     Precision vs Recall bubbles")
    print("  7  confusion_matrix.png      What gets confused with what")
    print("  8  example_annotation.png    Coloured CV token prediction")
    print("=" * 55)


# =============================================================================
# Main
# =============================================================================

def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.fp16 and torch.cuda.is_available()

    print(f"\n{'='*60}\nHireSense AI — Local Training\n"
          f"Device: {device}  FP16: {use_amp}\n{'='*60}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    all_ex = (load_resume_data(args.data_path)
              if args.data_path and os.path.exists(args.data_path) else [])
    n_synth = max(300, 310 - len(all_ex))
    all_ex += generate_synthetic(n_synth)

    train_ex, temp    = train_test_split(all_ex, test_size=0.2, random_state=42)
    val_ex,   test_ex = train_test_split(temp,   test_size=0.5, random_state=42)
    print(f"Train:{len(train_ex)}  Val:{len(val_ex)}  Test:{len(test_ex)}")

    nw = 0 if platform.system() == "Windows" else 4
    def mkloader(ex, bs, sh):
        ds = ResumeNERDataset(ex, tokenizer, args.max_length)
        return DataLoader(ds, batch_size=bs, shuffle=sh,
                          num_workers=nw, pin_memory=True)
    trl = mkloader(train_ex, args.batch_size,     True)
    vl  = mkloader(val_ex,   args.batch_size * 2, False)
    tsl = mkloader(test_ex,  args.batch_size * 2, False)

    configs = [("BERT_BiLSTM_CRF", args.lstm_layers)]
    if args.compare:
        configs.append(("BERT_CRF", 0))

    all_histories = {}

    for model_name, n_lstm in configs:
        print(f"\n{'='*55}\n  Training Model: {model_name}\n{'='*55}")
        model = BertBiLSTMCRF(
            "bert-base-uncased", args.lstm_hidden, n_lstm, 0.3, args.freeze_bert
        ).to(device)
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

        bp  = [p for n,p in model.named_parameters() if "bert"     in n and p.requires_grad]
        op  = [p for n,p in model.named_parameters() if "bert" not in n and p.requires_grad]
        opt = AdamW([{"params": bp, "lr": args.bert_lr},
                     {"params": op, "lr": args.lstm_lr}], weight_decay=0.01)

        total_steps  = len(trl) * args.epochs
        warmup_steps = int(total_steps * 0.1)
        sch          = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
        scaler       = GradScaler() if use_amp else None

        current_out_dir = os.path.join(args.output_dir, model_name) if args.compare else args.output_dir
        os.makedirs(current_out_dir, exist_ok=True)
        best_f1    = 0.0
        best_epoch = 1
        patience   = 0
        history: List[dict] = []

        for epoch in range(1, args.epochs + 1):
            model.train(); epoch_loss = 0.0; opt.zero_grad()
            bar = tqdm(trl, desc=f"Epoch {epoch}/{args.epochs}")
            for step, batch in enumerate(bar):
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["labels"].to(device)
                if use_amp:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        out  = model(ids, mask, lbls)
                        loss = out["loss"] / args.accumulation_steps
                    scaler.scale(loss).backward()
                    if (step+1) % args.accumulation_steps == 0 or (step+1) == len(trl):
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        prev = scaler.get_scale()
                        scaler.step(opt); scaler.update()
                        if scaler.get_scale() >= prev: sch.step()
                        opt.zero_grad()
                else:
                    out  = model(ids, mask, lbls)
                    loss = out["loss"] / args.accumulation_steps
                    loss.backward()
                    if (step+1) % args.accumulation_steps == 0 or (step+1) == len(trl):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step(); sch.step(); opt.zero_grad()
                epoch_loss += loss.item() * args.accumulation_steps
                bar.set_postfix({"loss": f"{loss.item()*args.accumulation_steps:.4f}"})

            vres     = evaluate(model, vl, device)
            avg_loss = epoch_loss / len(trl)
            print(f"Epoch {epoch}: loss={avg_loss:.4f}  val_f1={vres['f1']:.4f}")
            history.append({"epoch": epoch, "train_loss": avg_loss,
                             "val_loss": vres["loss"], "val_f1": vres["f1"]})

            if vres["f1"] > best_f1:
                best_f1 = float(vres["f1"]); best_epoch = epoch; patience = 0
                torch.save(
                    {"model_state_dict": model.state_dict(),
                     "label2id": LABEL2ID, "id2label": ID2LABEL, "f1": best_f1},
                    os.path.join(current_out_dir, "best_model.pt"),
                )
                print(f"  ✓ Saved (F1={best_f1:.4f})")
            else:
                patience += 1
                if patience >= args.patience: print("Early stopping."); break

        all_histories[model_name] = history

        ckpt = torch.load(os.path.join(current_out_dir, "best_model.pt"), weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        tres = evaluate(model, tsl, device)
        print(f"\nTest F1: {tres['f1']:.4f}  Loss: {tres['loss']:.4f}")
        print(classification_report(tres["labels"], tres["preds"]))

        torch.save(
            {"model_state_dict": model.state_dict(),
             "label2id": LABEL2ID, "id2label": ID2LABEL, "f1": float(tres["f1"])},
            os.path.join(current_out_dir, "resume_ner_model.pt"),
        )
        tokenizer.save_pretrained(os.path.join(current_out_dir, "tokenizer"))
        with open(os.path.join(current_out_dir, "model_config.json"), "w") as f:
            json.dump({"bert_model_name": "bert-base-uncased",
                       "lstm_hidden_size": args.lstm_hidden, "lstm_num_layers": n_lstm,
                       "num_labels": NUM_LABELS, "label2id": LABEL2ID,
                       "id2label": {str(k): v for k, v in ID2LABEL.items()}}, f, indent=2)

        print(f"Best Val F1: {best_f1:.4f}  |  Test F1: {tres['f1']:.4f}")
        print(f"Saved to: {current_out_dir}")

        # ── Generate all 8 plots ─────────────────────────────────────────────────
        generate_all_plots(
            history        = history,
            best_epoch     = best_epoch,
            total_steps    = total_steps,
            warmup_steps   = warmup_steps,
            bert_lr        = args.bert_lr,
            lstm_lr        = args.lstm_lr,
            train_examples = train_ex,
            test_true      = tres["labels"],
            test_pred      = tres["preds"],
            model          = model,
            tokenizer      = tokenizer,
            device         = str(device),
            output_dir     = current_out_dir,
        )

    if args.compare and len(all_histories) > 1:
        print(f"\n{'='*55}\n  Generating Comparative Analysis Plots\n{'='*55}")
        plot_comparative_f1(all_histories, args.output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",          type=str,   default="./data/kaggle_resume_pdf")
    p.add_argument("--output_dir",          type=str,   default="./output")
    p.add_argument("--batch_size",          type=int,   default=8)
    p.add_argument("--accumulation_steps",  type=int,   default=4)
    p.add_argument("--epochs",              type=int,   default=6)
    p.add_argument("--max_length",          type=int,   default=512)
    p.add_argument("--lstm_hidden",         type=int,   default=256)
    p.add_argument("--bert_lr",             type=float, default=2e-5)
    p.add_argument("--lstm_lr",             type=float, default=1e-3)
    p.add_argument("--patience",            type=int,   default=5)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--fp16",                action="store_true")
    p.add_argument("--freeze_bert",         action="store_true")
    p.add_argument("--compare",             action="store_true", help="Train both BERT-BiLSTM-CRF and BERT-CRF models to compare performance")
    p.add_argument("--lstm_layers",         type=int,   default=2)
    main(p.parse_args())

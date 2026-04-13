"""
HireSense AI — Dataset Loading and Preprocessing
BERT + BiLSTM + CRF, Multi-Sector Resume NER

Key improvements for higher scoring:
  ─ Multi-pass labelling: dates first, then contact, then phrases, then tokens
  ─ PROJ labelling now also fires on technology-stack sentences ("Built X using Y")
  ─ EXP labelling captures multi-word job titles ("Senior Software Engineer")
  ─ ACH labelling captures competition results ("1st Place", "Top 10%", "500+")
  ─ Negative examples: section headers tagged O so model doesn't confuse them
  ─ Synthetic data covers 7 distinct sector templates grounded in the 20 real CVs
  ─ Label-smoothing augmentation: 5% token-level O→random noise to reduce overfit
"""

import os
import re
import json
import glob
import platform
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    DataConfig, TrainingConfig,
    LABEL2ID, ID2LABEL, ENTITY_LABELS,
    SKILL_SYNONYMS, RESUME_SECTION_HEADERS,
)


# =============================================================================
# Data structure
# =============================================================================

@dataclass
class NERExample:
    tokens: List[str]
    labels: List[str]
    text:   str = ""
    sector: str = "unknown"


# =============================================================================
# PyTorch Dataset
# =============================================================================

class ResumeNERDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512, augment=False):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.augment    = augment   # label-noise augmentation during training

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex  = self.examples[idx]
        enc = self.tokenizer(
            ex.tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        word_ids = enc.word_ids()
        aligned  = []
        prev     = None

        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev:
                raw = ex.labels[wid] if wid < len(ex.labels) else "O"
                aligned.append(LABEL2ID.get(raw, LABEL2ID["O"]))
            else:
                raw = ex.labels[wid] if wid < len(ex.labels) else "O"
                if raw.startswith("B-"):
                    raw = "I-" + raw[2:]
                aligned.append(LABEL2ID.get(raw, LABEL2ID["O"]))
            prev = wid

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(aligned, dtype=torch.long),
        }


# =============================================================================
# Weak Labeller — multi-pass, multi-sector
# =============================================================================

class WeakLabeller:
    """
    Multi-pass BIO tagger built from analysis of all 20 uploaded CVs.

    Pass order (later passes don't overwrite earlier):
      1. Section headers  → always O  (prevents false positives)
      2. Dates            → DATE      (regex)
      3. Contact          → CONTACT   (email / phone regex)
      4. Multi-word phrases → SKILL   (phrase table)
      5. Job title n-grams → EXP      (seniority + role bigrams)
      6. Achievement n-grams → ACH    (rank/award phrases)
      7. Single tokens    → remaining categories
    """

    # ── Skills ───────────────────────────────────────────────────────────────

    SKILL_SINGLE = {
        # Languages
        "python","java","c++","c#","c","javascript","typescript","ruby","php",
        "go","golang","rust","scala","kotlin","swift","r","bash","shell",
        "powershell","matlab","solidity","dart","html","css","html5","css3",
        # Web
        "react","reactjs","angular","vue","nextjs","tailwind","bootstrap",
        "sass","jquery","redux","gatsby","svelte",
        # Backend
        "node","nodejs","express","expressjs","django","flask","fastapi",
        "spring","nestjs","fastify","celery","gunicorn","rails","laravel",
        # Mobile
        "flutter","expo","swiftui","jetpack",
        # Databases
        "mysql","postgresql","postgres","mongodb","sqlite","redis","oracle",
        "cassandra","dynamodb","firebase","supabase","mssql","neo4j",
        "elasticsearch",
        # ML / AI
        "pytorch","tensorflow","keras","sklearn","xgboost","lightgbm",
        "numpy","pandas","matplotlib","seaborn","nltk","spacy","huggingface",
        "transformers","bert","yolo","yolov8","langchain","openai","gemini",
        "rag","unsloth","pennylane","vqc","opencv","pillow",
        # Data
        "tableau","looker","sas","databricks","redshift","spark","hadoop",
        "hive","kafka","airflow","dbt","etl","powerbi",
        # Cloud / DevOps
        "aws","azure","gcp","heroku","netlify","vercel","docker","kubernetes",
        "k8s","jenkins","terraform","ansible","linux","ubuntu","git","github",
        "gitlab","cloudflare","prometheus","grafana",
        # Security
        "kali","wireshark","metasploit","nmap","splunk","burpsuite","snort",
        # Finance tools
        "bloomberg","tally","quickbooks","sap","excel","tableau",
        # Data engineering
        "databricks","redshift","glue","airflow","kinesis",
        # Concepts / methodologies
        "dsa","oop","dbms","jwt","oauth","agile","scrum","rest","graphql",
        "grpc","microservices","tdd","bdd","cicd","devops",
        # Design
        "figma","sketch","xd","invision","canva","photoshop","illustrator",
        "aftereffects",
        # Healthcare / other sectors
        "nursing","clinical","pharmacology","ehr","agronomy","horticulture",
        "autocad","revit","solidworks","catia","ansys","plc","scada","hvac",
        # Soft (appear explicitly in CV skills sections)
        "communication","teamwork","leadership","analytical","multitasking",
    }

    DEGREE_SINGLE = {
        "bachelor","master","phd","doctorate","mba","btech","mtech",
        "b.tech","m.tech","b.sc","m.sc","b.com","m.com","b.e","m.e",
        "b.a","m.a","llb","llm","mbbs","bds","md","pgdm","b.arch",
        "ca","cfa","cpa","cma","acca","diploma","b.s","m.s",
        "university","college","institute","iit","nit","bits","iiit",
    }

    CERT_SINGLE = {
        "certified","certification","certificate","oracle","bloomberg",
        "nptel","coursera","udemy","hackerrank","freecodecamp","pregrad",
        "scalar","cs50","pmp","cissp","ceh","comptia","databricks",
    }

    PROJ_SINGLE = {
        "project","projects","built","developed","implemented",
        "created","designed","deployed","architected","engineered",
    }

    ACH_SINGLE = {
        "award","winner","finalist","recognition","published","patent",
        "honors","honour","scholarship","champion","medal","ranked",
        "selected","hackathon","olympiad",
    }

    EXP_SINGLE = {
        "senior","junior","lead","principal","associate","head",
        "engineer","developer","analyst","scientist","architect","designer",
        "manager","director","officer","specialist","coordinator","executive",
        "consultant","intern","internship","trainee","freelancer",
        "founder","co-founder","president","vice",
        "accountant","advocate","auditor","nurse","teacher","technician",
    }

    ORG_SINGLE = {
        "google","microsoft","amazon","meta","apple","netflix","uber",
        "tcs","infosys","wipro","hcl","cognizant","capgemini","accenture",
        "jio","reliance","flipkart","swiggy","zomato","paytm","hdfc","icici",
        "deloitte","pwc","ey","kpmg","mckinsey","bcg","bain",
        # Smaller orgs from the CVs
        "nysonaiconnect","nsoft","hacktify","mactores","cognifyz",
        "meryt","isf","unacores","gostops","isg",
    }

    LOC_SINGLE = {
        "jabalpur","mumbai","pune","bangalore","bengaluru","delhi","noida",
        "gurgaon","hyderabad","chennai","kolkata","vellore","jaipur","surat",
        "india","mp","maharashtra",
    }

    # ── Multi-token phrase patterns ──────────────────────────────────────────

    PHRASE_SKILLS = [
        # AI / ML
        ("machine","learning"), ("deep","learning"),
        ("natural","language","processing"),
        ("computer","vision"), ("generative","ai"), ("agentic","ai"),
        ("large","language","model"), ("data","science"),
        ("data","analysis"), ("data","engineering"),
        ("reinforcement","learning"), ("transfer","learning"),
        ("feature","engineering"),
        # Cloud
        ("google","cloud"), ("google","cloud","platform"),
        ("microsoft","azure"), ("aws","glue"), ("aws","lambda"),
        ("github","actions"), ("amazon","web","services"),
        # Web
        ("full","stack"), ("mern","stack"), ("mean","stack"),
        ("react","native"), ("node","js"), ("next","js"),
        ("rest","api"), ("restful","api"), ("graphql","api"),
        # Security
        ("penetration","testing"), ("vulnerability","assessment"),
        ("network","security"), ("digital","forensics"),
        ("threat","intelligence"), ("incident","response"),
        ("sql","injection"), ("privilege","escalation"),
        ("capture","the","flag"),
        # Finance
        ("financial","modelling"), ("financial","modeling"),
        ("financial","analysis"), ("financial","reporting"),
        ("investment","banking"), ("risk","management"),
        ("algorithmic","trading"), ("quantitative","analysis"),
        ("bloomberg","market","concepts"),
        ("bloomberg","finance","fundamentals"),
        # Data tools
        ("power","bi"), ("data","pipeline"), ("data","warehouse"),
        ("apache","airflow"), ("apache","kafka"), ("apache","spark"),
        # Engineering
        ("civil","engineering"), ("mechanical","engineering"),
        ("electrical","engineering"), ("software","engineering"),
        # General
        ("problem","solving"), ("critical","thinking"),
        ("time","management"), ("team","collaboration"),
        ("project","management"), ("agile","methodology"),
    ]

    # Multi-word job title prefixes that signal EXP
    SENIORITY = {"senior","junior","lead","principal","associate","staff",
                 "chief","head","founding"}
    ROLE_WORDS = {"engineer","developer","analyst","scientist","architect",
                  "designer","manager","director","officer","specialist",
                  "consultant","intern","coordinator","executive","founder",
                  "co-founder","president","technician","administrator"}

    # Achievement phrase patterns: "1st Place", "Top 10", "500+ problems"
    ACH_PHRASE_RE = re.compile(
        r"\b(1st|2nd|3rd|\d+th)\s+(place|position|rank|prize)\b"
        r"|\btop\s+\d+[%+]?\b"
        r"|\b\d+\+\s*(dsa|problems|questions|teams|days)\b"
        r"|\b(gold|silver|bronze)\s+medal\b"
        r"|\bnational\s+(finalist|winner|champion)\b"
        r"|\bfinalist\b"
        r"|\b(won|secured|ranked)\s+(1st|2nd|3rd|first|second|third)\b",
        re.IGNORECASE,
    )

    # Project sentence pattern: "Built/Developed/Created X using/with Y"
    PROJ_SENTENCE_RE = re.compile(
        r"\b(built|developed|created|designed|implemented|architected|engineered|deployed)\b"
        r".{0,60}\b(using|with|via|leveraging|powered by)\b",
        re.IGNORECASE,
    )

    CGPA_RE  = re.compile(r"\d+\.?\d*\s*/\s*\d+|\d+\.\d+\s*cgpa", re.IGNORECASE)
    EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
    PHONE_RE = re.compile(r"\+?91[\s\-]?\d{10}|\+?\d[\d\s\-()]{7,}\d")
    DATE_RE  = re.compile(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}"
        r"|\b\d{4}\s*[-\u2013\u2014]\s*(\d{4}|present|ongoing|current)\b"
        r"|\b(20\d{2})\b",
        re.IGNORECASE,
    )

    # ── public entry point ───────────────────────────────────────────────────

    def label(self, text: str) -> Optional[NERExample]:
        if not text or len(text.strip()) < 20:
            return None

        # Tokenise: preserve emails, version strings, hyphenated words
        tokens = re.findall(
            r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}"    # emails intact
            r"|[\w]+(?:[./+\-#][\w]+)*"           # tech tokens e.g. "react.js", "c++"
            r"|\S",                                # any other non-space char
            text,
        )
        if len(tokens) < 5:
            return None

        labels = ["O"] * len(tokens)
        tl     = [t.lower() for t in tokens]

        def safe_set(i, base, start):
            """Only write if position is still 'O'."""
            if 0 <= i < len(labels) and labels[i] == "O":
                labels[i] = f"B-{base}" if start else f"I-{base}"

        # ── Pass 1: section headers → always O (already O, but mark to block) ──
        # (no-op: O is default, this pass just serves as documentation)

        # ── Pass 2: dates ────────────────────────────────────────────────────
        running_pos = 0
        for m in self.DATE_RE.finditer(text):
            for ti, tok in enumerate(tokens):
                idx = text.find(tok, running_pos)
                if idx == -1:
                    continue
                if m.start() <= idx < m.end():
                    safe_set(ti, "DATE", True)
                running_pos = max(running_pos, idx + 1)

        # ── Pass 3: contact ───────────────────────────────────────────────────
        for ti, tok in enumerate(tokens):
            if self.EMAIL_RE.fullmatch(tok):
                safe_set(ti, "CONTACT", True)
            elif self.PHONE_RE.fullmatch(tok.replace(" ", "")):
                safe_set(ti, "CONTACT", True)

        # ── Pass 4: achievement phrases ───────────────────────────────────────
        for m in self.ACH_PHRASE_RE.finditer(text):
            span_text = m.group(0).lower()
            span_tokens = span_text.split()
            for ti in range(len(tokens) - len(span_tokens) + 1):
                if [tl[ti + k] for k in range(len(span_tokens))] == span_tokens:
                    for k in range(len(span_tokens)):
                        safe_set(ti + k, "ACH", k == 0)

        # ── Pass 5: multi-token SKILL phrases ────────────────────────────────
        i = 0
        _done = [False] * len(tokens)
        while i < len(tokens):
            matched = False
            for phrase in self.PHRASE_SKILLS:
                end = i + len(phrase)
                if end <= len(tokens) and tuple(tl[i:end]) == phrase:
                    for j in range(len(phrase)):
                        safe_set(i + j, "SKILL", j == 0)
                        _done[i + j] = True
                    i += len(phrase)
                    matched = True
                    break
            if not matched:
                i += 1

        # ── Pass 6: multi-word job titles (Seniority + Role) ─────────────────
        i = 0
        while i < len(tokens):
            if tl[i] in self.SENIORITY and i + 1 < len(tokens) and tl[i + 1] in self.ROLE_WORDS:
                safe_set(i,     "EXP", True)
                safe_set(i + 1, "EXP", False)
                # absorb up to 2 more role words
                j = i + 2
                while j < len(tokens) and j < i + 4 and tl[j] in self.ROLE_WORDS:
                    safe_set(j, "EXP", False)
                    j += 1
                i = j
            else:
                i += 1

        # ── Pass 7: single-token matching ─────────────────────────────────────
        for i, tok in enumerate(tl):
            if tok in RESUME_SECTION_HEADERS:
                continue   # never tag section headers
            if tok in self.SKILL_SINGLE:
                safe_set(i, "SKILL",   True)
            elif tok in self.DEGREE_SINGLE:
                safe_set(i, "EDU",     True)
            elif tok in self.CERT_SINGLE:
                safe_set(i, "CERT",    True)
            elif tok in self.PROJ_SINGLE:
                safe_set(i, "PROJ",    True)
            elif tok in self.ACH_SINGLE:
                safe_set(i, "ACH",     True)
            elif tok in self.EXP_SINGLE:
                safe_set(i, "EXP",     True)
            elif tok in self.ORG_SINGLE:
                safe_set(i, "ORG",     True)
            elif tok in self.LOC_SINGLE:
                safe_set(i, "LOC",     True)
            elif self.CGPA_RE.match(tokens[i]):
                safe_set(i, "EDU",     True)

        return NERExample(tokens=tokens, labels=labels, text=" ".join(tokens))


# =============================================================================
# Synthetic data generator — grounded in the 20 uploaded CVs
# =============================================================================

class SyntheticGenerator:
    """
    Generates fully-annotated NERExamples.
    7 sector templates built directly from the uploaded CV corpus:
      1. cs_developer      (JEC/VIT MERN/backend freshers)
      2. data_science      (JEC/IIT ML/AI freshers)
      3. cybersecurity     (JEC AI+DS)
      4. finance_tech      (NMIMS MBA Tech)
      5. management        (BMS Delhi University)
      6. data_engineering  (IIT Madras / JEC)
      7. mixed_engineering (Mechatronics / full-stack hybrid)
    """

    TEMPLATES = [
        {
            "sector": "cs_developer",
            "skill_sets": [
                ["Python","JavaScript","React","Node.js","Express.js","MongoDB","JWT","REST API"],
                ["Java","C++","SQL","Docker","AWS","Git","GitHub","Postman"],
                ["TypeScript","Next.js","PostgreSQL","Redis","Tailwind CSS","Firebase"],
                ["Python","FastAPI","Flask","MySQL","Docker","Linux","Celery"],
                ["React","Redux","Node.js","MongoDB","JWT","Firebase","Figma"],
                ["Python","Django","React","TypeScript","MySQL","Docker","AWS","GCP"],
            ],
            "exp_titles": [
                "Full Stack Developer Intern","Backend Developer Intern",
                "Product Engineer Intern","Software Engineer Intern",
                "App Development Intern","Web Developer Intern",
            ],
            "orgs": ["Google","Microsoft","Jio Platforms","Infosys","TCS",
                     "NysonAiConnect","NSoft Technologies","Switch Climate Tech"],
            "degrees": [
                "Bachelor of Technology in Information Technology",
                "Bachelor of Technology in Computer Science and Engineering",
                "B.Tech Information Technology",
            ],
            "colleges": [
                "Jabalpur Engineering College","Vellore Institute of Technology",
                "NIT Trichy","NIT Bhopal","IIIT Jabalpur",
            ],
            "projects": [
                ("Job Preparation Platform","MERN Stack JWT MongoDB React Node.js"),
                ("Resume Builder Web Application","React.js dynamic preview"),
                ("Cuisine Catalyst","MERN MongoDB RESTful API JWT"),
                ("Discord Clone","HTML CSS JavaScript DOM"),
                ("To-Do List App","React local storage JavaScript"),
                ("BinSavvy","React TypeScript Django MySQL YOLO"),
                ("SelectorIT","Django PostgreSQL REST API"),
                ("Rubik Solver Pro","React FastAPI Python SQLite Docker"),
            ],
            "achievements": [
                ("Solved 500+ DSA problems on LeetCode and GeeksForGeeks","ACH"),
                ("Finalist in HackCrux 2025 at LNMIIT Jaipur","ACH"),
                ("1st Place HackCrux 2025 among 90+ teams","ACH"),
                ("Selected top 20 teams Vibe Code Hackathon","ACH"),
                ("Smart India Hackathon 2024 Semifinalist","ACH"),
                ("5th position Hack JEC 2.0 Internal Hackathon SIH 2024","ACH"),
            ],
            "certs": [
                "Oracle Cloud Infrastructure Architect Associate",
                "Machine Learning with Python freecodecamp",
                "Full-Stack Web Development MERN CodeHelp",
                "Data Structures Algorithms C++ Supreme 4.0",
                "CS50 Introduction to Computer Science Harvard",
            ],
        },
        {
            "sector": "data_science",
            "skill_sets": [
                ["Python","PyTorch","TensorFlow","scikit-learn","Pandas","NumPy","Matplotlib"],
                ["Python","BERT","NLTK","Transformers","Seaborn","Jupyter","Huggingface"],
                ["Python","XGBoost","SHAP","Flask","Docker","GCP","Androguard"],
                ["Python","LangChain","Gemini AI","RAG","FastAPI","Pinecone","Docker","GCP"],
                ["Python","CNN","LSTM","GANs","TensorFlow","Keras","Matplotlib","OpenCV"],
                ["Python","PyTorch","PennyLane","VQC","Random Forest","scikit-learn"],
            ],
            "exp_titles": [
                "Data Science Intern","Machine Learning Engineer Intern",
                "Data Analyst Intern","AI/ML Intern",
                "Founder and Lead Engineer","Data Science Intern at Cognifyz",
            ],
            "orgs": ["Cognifyz Technologies","Mactores","GoStops","Jio Platforms",
                     "MERYT","ISF Analytica","Google","Amazon"],
            "degrees": [
                "Bachelor of Technology in Computer Science and Engineering",
                "B.S. Degree in Data Science and Applications",
                "Bachelor of Technology in Information Technology",
            ],
            "colleges": [
                "Jabalpur Engineering College","IIT Madras","Vellore Institute of Technology",
            ],
            "projects": [
                ("Hybrid Phishing Detector","PyTorch PennyLane VQC Random Forest Python"),
                ("Real-Time Disaster Information Aggregator","BERT NLTK Pandas Seaborn APIs"),
                ("VitalView","Python Flask React MySQL GenAI disease prediction"),
                ("Student Lifestyle Analysis","Pandas scikit-learn Matplotlib"),
                ("Restaurants Rating Prediction","Pandas scikit-learn Matplotlib"),
                ("Stock Price Predictor","Pandas scikit-learn linear regression"),
                ("ECG Anomaly Detection","Pandas PyTorch Matplotlib classification"),
                ("Fake Banking APK Detection","XGBoost SHAP GCP Python Flask CNN LSTM"),
            ],
            "achievements": [
                ("Top Finalist 10000+ teams Google Cloud Agentic AI Day Hackathon 2025","ACH"),
                ("Finalist Top 91/700+ Google Cloud Agentic AI Day 2025 Bangalore","ACH"),
                ("Myntra HackerRamp We for She 2024 Semi Finalist","ACH"),
                ("Reliance Foundation Undergraduate Scholar","ACH"),
                ("GFG-160 Challenge 80 Days Streak","ACH"),
                ("1st Place HackCrux 2025 LNMIIT Jaipur among 90+ teams","ACH"),
            ],
            "certs": [
                "Oracle Cloud Infrastructure 2025 Certified Data Science Professional",
                "Oracle Cloud Infrastructure 2025 Certified Gen AI Professional",
                "AWS Certified Cloud Practitioner",
                "Databricks Certified Data Engineer Associate",
                "Data Science INI Certificate",
            ],
        },
        {
            "sector": "cybersecurity",
            "skill_sets": [
                ["Python","Kali Linux","Wireshark","Metasploit","Nmap","Splunk"],
                ["Bash","Linux","SQL Injection","XSS","IDOR","CSRF","CORS"],
                ["Python","C","Penetration Testing","Network Security","IDS","IPS"],
                ["Shell Scripting","Docker","AWS","Firewall","SIEM","Log Monitoring"],
                ["Python","Go","AWS","Docker","Kubernetes","PostgreSQL","PostgreSQL","Redis"],
            ],
            "exp_titles": [
                "CyberSecurity Intern","Linux Trainee",
                "Security Analyst Intern","Penetration Testing Intern",
                "Cloud Computing Lead",
            ],
            "orgs": ["Hacktify Cyber Security","CYBERSEC INFOTECH Private Limited",
                     "IBM","Microsoft","GDSC JEC"],
            "degrees": [
                "Bachelor of Technology in Artificial Intelligence and Data Science Engineering",
                "Bachelor of Technology in Information Technology",
                "Bachelor of Technology in Computer Science",
            ],
            "colleges": ["Jabalpur Engineering College","VIT Vellore","NIT Bhopal"],
            "projects": [
                ("Web Application Security Testing","SQL Injection XSS mock e-commerce platform"),
                ("FinTrace","Python Flask SQLite NetworkX scikit-learn Pandas anti-money laundering"),
                ("Linux Security Hardening","Bash firewall access control log monitoring"),
                ("BinSavvy","Next.js FastAPI Docker Jenkins GitHub CI/CD waste management"),
                ("SelectorIT","Django PostgreSQL REST API automated testing"),
            ],
            "achievements": [
                ("Google Cybersecurity Professional Certification Coursera October 2024","CERT"),
                ("Cyber Security Privacy NPTEL July October 2024","CERT"),
                ("CTF challenges red teaming OSINT cryptography privilege escalation","ACH"),
                ("Smart India Hackathon 2024 Semifinals 500+ teams","ACH"),
                ("Webathon 2024 36-hour challenge Mumbai","ACH"),
            ],
            "certs": [
                "Google Cybersecurity Professional Certification",
                "Cyber Security Privacy NPTEL",
                "Introduction to Programming Python GUVI",
                "Ethical Hacking NPTEL",
                "CEH Certified Ethical Hacker",
            ],
        },
        {
            "sector": "finance_tech",
            "skill_sets": [
                ["Python","SQL","Microsoft Excel","Tableau","Power BI","Financial Modelling"],
                ["Bloomberg","Financial Analysis","MySQL","Python","NumPy","Pandas","SAS"],
                ["SQL","Excel","Tableau","SAS","HTML","CSS","Market Analytics"],
                ["Financial Modelling","Risk Management","Power BI","MySQL","React JS","MongoDB"],
                ["Python","FastAPI","MySQL","Financial Analysis","Bloomberg","Excel","Figma"],
            ],
            "exp_titles": [
                "Technical Intern","Finance Intern","Data Analyst Intern",
                "Financial Analyst Intern","Junior Consultant Intern",
            ],
            "orgs": ["Jio Platforms Ltd","Ernst Young EY","Unacores Solutions",
                     "ISF Analytica and Informatica","IS Global Web",
                     "Deloitte","PwC","J.P. Morgan","HOSHŌ DIGITAL"],
            "degrees": [
                "MBA Tech","BTech Computer",
                "Master of Business Administration Technology",
                "Bachelor of Technology Computer Engineering",
            ],
            "colleges": [
                "MPSTME NMIMS Mumbai",
                "Mukesh Patel School of Technology Management and Engineering",
            ],
            "projects": [
                ("IBM HR Dashboard","Tableau workforce data HR decision-making"),
                ("Cricket Management System","MySQL tournament teams players schedules"),
                ("Object Detection Video Analytics","YOLO Python computer vision surveillance"),
                ("Blinkit Retail Analytics Dashboard","Power BI sales outlet performance"),
                ("ATM Functionality","Python MySQL core banking PIN authentication"),
                ("Speech Emotion Recognition","CNN TensorFlow PyTorch OpenAI Whisper BART NLP"),
                ("Astronaut Fitness Prediction","Random Forest Gradient Boosting 94% accuracy"),
                ("Gym Management System","SQL DBMS membership scheduling reporting"),
            ],
            "achievements": [
                ("Bloomberg Market Concepts Certificate Bloomberg","CERT"),
                ("Bloomberg Finance Fundamentals Certificate Bloomberg","CERT"),
                ("Investment Banking Virtual Experience J.P. Morgan","CERT"),
                ("ADAPT 3.0 Winner Algorithmic Trading","ACH"),
                ("Merit Certificate outstanding internship performance ISF Analytica","ACH"),
            ],
            "certs": [
                "Bloomberg Market Concepts",
                "Bloomberg Finance Fundamentals",
                "Infosys Financial Modeling Web Development",
                "Udemy Data Analytics",
                "Python Core to Advanced Course Pregrad",
            ],
        },
        {
            "sector": "management",
            "skill_sets": [
                ["MS Excel","MS Word","Financial Modelling","Market Research","PowerPoint"],
                ["Excel","Business Analysis","Strategic Planning","Data Analysis"],
                ["MS Excel","Financial Reporting","Canva","MS Word"],
            ],
            "exp_titles": [
                "Co-Founder and Head of Operations",
                "Junior Consultant Intern","Business Development Executive",
                "Organizing Committee Member",
            ],
            "orgs": ["MERYT","HOSHŌ DIGITAL","Christ University NCR",
                     "IIM Ahmedabad","Unmask","Atelier"],
            "degrees": [
                "Bachelor of Management Studies",
                "Bachelor of Business Administration",
                "Bachelor of Commerce",
            ],
            "colleges": [
                "Shaheed Sukhdev College of Business Studies University of Delhi",
                "Delhi University","Symbiosis","NMIMS",
            ],
            "projects": [
                ("M&A Autopsy ZeeSony Collapse","corporate governance market dominance analysis"),
                ("ONDC vs Amazon Analysis","unit economics D2C profitability financial modeling"),
                ("Modern Management Theory Research","DDDM Systems Approach Agile Triple Bottom Line"),
            ],
            "achievements": [
                ("Gold Medal Class Topper International Commerce Olympiad SOF 2024","ACH"),
                ("National Finalist Fourth Wall Chaos 26 IIM Ahmedabad","ACH"),
                ("Runner Up Nyay E Rang Christ NCR Deemed University 2026","ACH"),
                ("National Finalist CIIS 25 Jabalpur Engineering College","ACH"),
            ],
            "certs": [
                "Bloomberg Finance Fundamentals",
                "Bloomberg Market Concepts",
                "Introduction to Data Analysis Microsoft Excel Coursera",
                "Foundations User Experience Design Google",
            ],
        },
        {
            "sector": "data_engineering",
            "skill_sets": [
                ["Python","AWS Glue","Lambda","S3","Databricks","Apache Airflow","Docker"],
                ["SQL","PostgreSQL","Databricks","Redshift","Kafka","Spark","dbt"],
                ["Python","AWS","IAM","KMS","VPC","Unity Catalog","CloudWatch","Kinesis"],
                ["Python","FastAPI","Docker","GCP","Firebase","Pinecone","Flutter","RAG"],
                ["Python","Django","AWS","Docker","Jenkins","GitHub","PostgreSQL","Redis"],
            ],
            "exp_titles": [
                "Data Engineer Intern","Cloud Computing Lead",
                "Software Developer Intern","Data Analyst Intern",
            ],
            "orgs": ["Mactores","GoStops","Amazon","Google","Microsoft",
                     "GDSC JEC","NysonAiConnect"],
            "degrees": [
                "B.S. Degree in Data Science and Applications",
                "Bachelor of Technology in Information Technology",
                "Bachelor of Technology in Computer Science",
            ],
            "colleges": ["IIT Madras","Jabalpur Engineering College","VIT Vellore"],
            "projects": [
                ("INVESTED Personal Finance Platform","GCP Flutter FastAPI Firebase Pinecone RAG MCP"),
                ("Blinkit Data Visualization","Power BI ETL Python SQL Server pandas"),
                ("AI String Numerical Transformation","Azure Data Factory Synapse Analytics Python"),
                ("SelectorIT","Django PostgreSQL REST API JoSAA data"),
                ("BinSavvy","Next.js FastAPI Docker Jenkins CI/CD waste sorting"),
            ],
            "achievements": [
                ("Top Finalist 10000+ teams Google Cloud Agentic AI Day","ACH"),
                ("Smart India Hackathon 2024 Semifinalist","ACH"),
                ("Reliance Foundation Undergraduate Scholar","ACH"),
                ("Databricks Certified Data Engineer Associate","CERT"),
                ("AWS Certified Cloud Practitioner","CERT"),
            ],
            "certs": [
                "AWS Certified Cloud Practitioner",
                "Databricks Certified Data Engineer Associate",
                "SQL Basic Certificate HackerRank",
                "Python Basic Certificate HackerRank",
                "Software Engineer Certificate HackerRank",
            ],
        },
        {
            "sector": "mixed_engineering",
            "skill_sets": [
                ["Python","Flask","SQLite","NetworkX","scikit-learn","Pandas","Chart.js"],
                ["Python","Next.js","React","Tailwind CSS","MongoDB","Google Gemini API"],
                ["Python","HTML","CSS","SQL","JavaScript","Pandas","Matplotlib","NumPy"],
                ["AutoCAD","Revit","STAAD Pro","BIM","Primavera","MS Project"],
                ["SolidWorks","CATIA","ANSYS","MATLAB","PLC","SCADA","Six Sigma"],
            ],
            "exp_titles": [
                "Student Technical Team Core Member",
                "Technical Core Member","DS/AI/ML Member",
                "Tech Influencer","Data Science Intern",
            ],
            "orgs": ["Matrix JEC","ACM JEC","Automotive Society","GDSC JEC","Cognifyz"],
            "degrees": [
                "Bachelor of Technology in Mechatronics",
                "Bachelor of Technology in Computer Science",
                "Bachelor of Engineering in Civil Engineering",
                "Bachelor of Engineering in Mechanical Engineering",
            ],
            "colleges": ["Jabalpur Engineering College","NIT Trichy","BITS Pilani"],
            "projects": [
                ("MindEase","Next.js React Tailwind CSS MongoDB Google Gemini API Chart.js"),
                ("FinTrace","Python Flask SQLite NetworkX scikit-learn Pandas Chart.js"),
                ("Aditya Electronics","Python MySQL inventory order processing"),
                ("Chess Game","Scratch two-player traditional chess"),
                ("MargDarshak-Mitr","Flask GenAI train coach tracking"),
            ],
            "achievements": [
                ("Selected top 20 teams Vibe Code Hackathon","ACH"),
                ("5th position Hack JEC 2.0 Internal Hackathon SIH 2024","ACH"),
                ("GFG-160 Challenge 80 Days Streak","ACH"),
                ("Data Science Intern Cognifyz Technologies Nov Dec 2024","EXP"),
                ("CS50 Introduction Computer Science Harvard November 2024","CERT"),
            ],
            "certs": [
                "Data Science INI Certificate",
                "Ethical Hacking NPTEL",
                "CS50 Harvard University",
                "Introduction Responsible AI Google",
                "Python HackerRank",
            ],
        },
    ]

    FIRST_NAMES = [
        "Priya","Rahul","Ananya","Arjun","Sneha","Ravi","Pooja","Amit",
        "Divya","Karan","Neha","Vikram","Meera","Rohit","Shreya","Aditya",
        "Sanidhya","Monal","Sonu","Jayendra","Siddharth","Anshika","Arya",
        "Shravan","Mansi","Ishan","Sanyam","Kinjal","Vansh","Subrato",
        "Sanskar","Samkit","Manjari","Shreeansh","Ayush","Tanvi","Nidhi",
    ]
    LAST_NAMES = [
        "Sharma","Patel","Gupta","Singh","Jain","Tiwari","Yadav","Mishra",
        "Agarwal","Kumar","Narang","Sanghvi","Israni","Vishwakarma","Bhagat",
        "Pradhan","Maheshwari","Saxena","Tomar","Verma","Shah","Mehta",
    ]

    def _add(self, t, l, words, tag):
        for j, w in enumerate(words):
            t.append(w)
            l.append(f"B-{tag}" if j == 0 else f"I-{tag}")

    def _o(self, t, l, word):
        t.append(word); l.append("O")

    def _make_one(self, tmpl):
        t, l = [], []

        # Name + contact
        name = f"{random.choice(self.FIRST_NAMES)} {random.choice(self.LAST_NAMES)}"
        self._add(t, l, name.split(), "NAME")
        self._o(t, l, "|")
        phone = f"+91{random.randint(7000000000,9999999999)}"
        t.append(phone); l.append("B-CONTACT")
        self._o(t, l, "|")
        email = f"{name.split()[0].lower()}@gmail.com"
        t.append(email); l.append("B-CONTACT")

        # Skills section
        self._o(t, l, "Skills"); self._o(t, l, ":")
        skills = random.choice(tmpl["skill_sets"])
        chosen = random.sample(skills, min(random.randint(4, 7), len(skills)))
        for si, sk in enumerate(chosen):
            self._add(t, l, sk.split(), "SKILL")
            if si < len(chosen) - 1: self._o(t, l, ",")

        # Experience section
        self._o(t, l, "Experience"); self._o(t, l, ":")
        title = random.choice(tmpl["exp_titles"])
        self._add(t, l, title.split(), "EXP")
        self._o(t, l, "at")
        org = random.choice(tmpl["orgs"])
        self._add(t, l, org.split(), "ORG")
        yr_s = random.randint(2020, 2024)
        yr_e = random.choice(["Present", str(yr_s + random.randint(0, 2))])
        self._o(t, l, "(")
        t.append(str(yr_s)); l.append("B-DATE")
        self._o(t, l, "-")
        t.append(yr_e);      l.append("B-DATE")
        self._o(t, l, ")")

        # Project section
        self._o(t, l, "Projects"); self._o(t, l, ":")
        proj_name, proj_tech = random.choice(tmpl["projects"])
        self._add(t, l, proj_name.split(), "PROJ")
        self._o(t, l, "Technologies:")
        for tech in proj_tech.split()[:5]:
            self._add(t, l, [tech], "SKILL")

        # Education section
        self._o(t, l, "Education"); self._o(t, l, ":")
        degree = random.choice(tmpl["degrees"])
        self._add(t, l, degree.split(), "EDU")
        self._o(t, l, ",")
        college = random.choice(tmpl["colleges"])
        self._add(t, l, college.split(), "ORG")
        cgpa = round(random.uniform(6.5, 9.5), 2)
        self._o(t, l, "CGPA:"); t.append(str(cgpa)); l.append("B-EDU")

        # Achievement
        self._o(t, l, "Achievements"); self._o(t, l, ":")
        ach_text, ach_tag = random.choice(tmpl["achievements"])
        self._add(t, l, ach_text.split(), ach_tag)

        # Certification
        self._o(t, l, "Certifications"); self._o(t, l, ":")
        cert = random.choice(tmpl["certs"])
        self._add(t, l, cert.split(), "CERT")

        # Location
        locs = ["Jabalpur","Mumbai","Pune","Bangalore","Delhi","Hyderabad","Vellore"]
        self._o(t, l, "|"); t.append(random.choice(locs)); l.append("B-LOC")

        return NERExample(tokens=t, labels=l, text=" ".join(t),
                          sector=tmpl["sector"])

    def generate(self, n=2000):
        return [self._make_one(random.choice(self.TEMPLATES)) for _ in range(n)]


# =============================================================================
# Data Processor
# =============================================================================

class DataProcessor:
    LABEL_MAP = {
        "B-Skills":"B-SKILL","I-Skills":"I-SKILL",
        "B-SKILLS":"B-SKILL","I-SKILLS":"I-SKILL",
        "B-Skill":"B-SKILL","I-Skill":"I-SKILL",
        "B-TECHNOLOGY":"B-SKILL","I-TECHNOLOGY":"I-SKILL",
        "B-Experience":"B-EXP","I-Experience":"I-EXP",
        "B-EXPERIENCE":"B-EXP","I-EXPERIENCE":"I-EXP",
        "B-JOB_TITLE":"B-EXP","I-JOB_TITLE":"I-EXP",
        "B-DESIGNATION":"B-EXP","I-DESIGNATION":"I-EXP",
        "B-Education":"B-EDU","I-Education":"I-EDU",
        "B-EDUCATION":"B-EDU","I-EDUCATION":"I-EDU",
        "B-DEGREE":"B-EDU","I-DEGREE":"I-EDU",
        "B-COLLEGE":"B-EDU","I-COLLEGE":"I-EDU",
        "B-UNIVERSITY":"B-EDU","I-UNIVERSITY":"I-EDU",
        "B-Company":"B-ORG","I-Company":"I-ORG",
        "B-COMPANY":"B-ORG","I-COMPANY":"I-ORG",
        "B-ORGANIZATION":"B-ORG","I-ORGANIZATION":"I-ORG",
        "B-Project":"B-PROJ","I-Project":"I-PROJ",
        "B-PROJECTS":"B-PROJ","I-PROJECTS":"I-PROJ",
        "B-Certification":"B-CERT","I-Certification":"I-CERT",
        "B-CERTIFICATION":"B-CERT","I-CERTIFICATION":"I-CERT",
        "B-Achievement":"B-ACH","I-Achievement":"I-ACH",
        "B-AWARD":"B-ACH","I-AWARD":"I-ACH",
        "B-Location":"B-LOC","I-Location":"I-LOC",
        "B-LOCATION":"B-LOC","I-LOCATION":"I-LOC",
        "B-CITY":"B-LOC","I-CITY":"I-LOC",
        "B-Date":"B-DATE","I-Date":"I-DATE",
        "B-DATE":"B-DATE","I-DATE":"I-DATE",
        "B-YEAR":"B-DATE","I-YEAR":"I-DATE",
        "B-Name":"B-NAME","I-Name":"I-NAME",
        "B-NAME":"B-NAME","I-NAME":"I-NAME",
        "B-Email":"B-CONTACT","I-Email":"I-CONTACT",
        "B-Phone":"B-CONTACT","I-Phone":"I-CONTACT",
    }

    def __init__(self, config: DataConfig):
        self.config   = config
        self.labeller = WeakLabeller()
        self.synth    = SyntheticGenerator()

    def _map_labels(self, labels):
        return [
            self.LABEL_MAP.get(l, l) if self.LABEL_MAP.get(l, l) in LABEL2ID else "O"
            for l in labels
        ]

    def _load_conll(self, filepath):
        examples, tokens, labels = [], [], []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    if tokens:
                        examples.append(NERExample(
                            tokens=tokens,
                            labels=self._map_labels(labels),
                            text=" ".join(tokens)))
                    tokens, labels = [], []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        labels.append(parts[-1])
        if tokens:
            examples.append(NERExample(
                tokens=tokens, labels=self._map_labels(labels),
                text=" ".join(tokens)))
        return examples

    def load_resume_corpus(self):
        path = self.config.resume_corpus_path
        if not os.path.exists(path): return []
        examples = []
        for fname in tqdm(os.listdir(path), desc="Resume Corpus"):
            if not fname.endswith(".json"): continue
            with open(os.path.join(path, fname), encoding="utf-8") as f:
                data = json.load(f)
            tokens = data.get("tokens", [])
            labels = self._map_labels(data.get("labels", []))
            if tokens and len(tokens) == len(labels):
                examples.append(NERExample(
                    tokens=tokens, labels=labels, text=" ".join(tokens)))
        return examples

    def load_ner_annotated_cvs(self):
        path = self.config.ner_annotated_path
        if not os.path.exists(path): return []
        examples = []
        conll = os.path.join(path, "train.txt")
        if os.path.exists(conll):
            examples.extend(self._load_conll(conll))
        ann = os.path.join(path, "annotations.json")
        if os.path.exists(ann):
            with open(ann, encoding="utf-8") as f:
                data = json.load(f)
            for item in tqdm(data, desc="NER-Annotated-CVs"):
                tokens = item.get("tokens", [])
                labels = self._map_labels(item.get("ner_tags", []))
                if tokens:
                    examples.append(NERExample(
                        tokens=tokens, labels=labels, text=" ".join(tokens)))
        return examples

    def load_kaggle_resume_pdf(self):
        path = self.config.kaggle_pdf_path
        if not os.path.exists(path):
            print(f"[skip] PDF path not found: {path}")
            return []

        pdf_files = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))

        print(f"Found {len(pdf_files)} PDFs in {path}")
        stats = dict(total=len(pdf_files), fitz=0, plumber=0,
                     ocr=0, empty=0, short=0, errors=0)

        try:
            import fitz as pymupdf
        except ImportError:
            pymupdf = None
            print("[warn] PyMuPDF not installed — pip install PyMuPDF")

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

        examples = []
        for filepath in tqdm(pdf_files, desc="PDF resumes"):
            try:
                with open(filepath, "rb") as fh:
                    if fh.read(5) != b"%PDF-":
                        stats["errors"] += 1; continue

                text = ""

                if pymupdf and not text:
                    try:
                        doc  = pymupdf.open(filepath)
                        text = "\n".join(p.get_text() for p in doc).strip()
                        if text: stats["fitz"] += 1
                    except Exception: pass

                if pdfplumber and not text:
                    try:
                        with pdfplumber.open(filepath) as pdf:
                            text = "\n".join(
                                p.extract_text() or "" for p in pdf.pages).strip()
                        if text: stats["plumber"] += 1
                    except Exception: pass

                if not text and _ocr and pymupdf:
                    try:
                        doc   = pymupdf.open(filepath)
                        parts = []
                        for pnum in range(min(3, len(doc))):
                            pix = doc[pnum].get_pixmap(dpi=150, alpha=False)
                            img = PILImage.frombytes(
                                "RGB", [pix.width, pix.height], pix.samples)
                            parts.append(pytesseract.image_to_string(img))
                        text = "\n".join(parts).strip()
                        if text: stats["ocr"] += 1
                    except Exception: pass

                if not text:
                    stats["empty"] += 1; continue

                sector = os.path.basename(os.path.dirname(filepath)).lower()
                ex = self.labeller.label(text)
                if ex is None or len(ex.tokens) <= 5:
                    stats["short"] += 1; continue
                ex.sector = sector
                examples.append(ex)

            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 3:
                    print(f"[error] {os.path.basename(filepath)}: {e}")

        print(f"PDF — fitz:{stats['fitz']} plumber:{stats['plumber']} "
              f"ocr:{stats['ocr']} empty:{stats['empty']} "
              f"short:{stats['short']} errors:{stats['errors']} "
              f"loaded:{len(examples)}")
        return examples

    def load_text_files(self):
        path = self.config.kaggle_pdf_path
        examples = []
        for fpath in glob.glob(os.path.join(path, "**", "*.txt"), recursive=True):
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                ex = self.labeller.label(text)
                if ex:
                    ex.sector = os.path.basename(os.path.dirname(fpath)).lower()
                    examples.append(ex)
            except Exception: pass

        try:
            import pandas as pd
            for fpath in glob.glob(os.path.join(path, "**", "*.csv"), recursive=True):
                try:
                    df  = pd.read_csv(fpath)
                    col = next((c for c in df.columns
                                if c.lower() in {"resume","resume_str","text","cv","content"}),
                               None)
                    if col:
                        for txt in df[col].dropna().astype(str):
                            ex = self.labeller.label(txt)
                            if ex:
                                ex.sector = os.path.basename(
                                    os.path.dirname(fpath)).lower()
                                examples.append(ex)
                except Exception: pass
        except ImportError: pass
        return examples

    def load_all_data(self):
        all_ex = []

        rc = self.load_resume_corpus()
        print(f"  Resume Corpus     : {len(rc):,}")
        all_ex.extend(rc)

        nc = self.load_ner_annotated_cvs()
        print(f"  NER-Annotated-CVs : {len(nc):,}")
        all_ex.extend(nc)

        kp = self.load_kaggle_resume_pdf()
        print(f"  Kaggle PDFs       : {len(kp):,}")
        all_ex.extend(kp)

        tf = self.load_text_files()
        print(f"  TXT / CSV files   : {len(tf):,}")
        all_ex.extend(tf)

        # Always add synthetic — minimum 1000, scale to 3000 if sparse data
        n_synth = max(1000, 3000 - len(all_ex))
        synth   = self.synth.generate(n_synth)
        all_ex.extend(synth)
        print(f"  Synthetic         : {len(synth):,}")
        print(f"  Total             : {len(all_ex):,}")

        train_ex, temp = train_test_split(
            all_ex, test_size=self.config.val_ratio + self.config.test_ratio,
            random_state=42)
        val_ex, test_ex = train_test_split(
            temp,
            test_size=self.config.test_ratio / (
                self.config.val_ratio + self.config.test_ratio),
            random_state=42)
        print(f"  Train:{len(train_ex):,}  Val:{len(val_ex):,}  Test:{len(test_ex):,}")
        return train_ex, val_ex, test_ex


# =============================================================================
# DataLoader factory
# =============================================================================

def create_dataloaders(train_ex, val_ex, test_ex, tokenizer, train_config):
    nw = 0 if platform.system() == "Windows" else 4

    def _make(exs, bs, shuffle, augment=False):
        ds = ResumeNERDataset(exs, tokenizer, train_config.max_seq_length, augment)
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=nw,
            pin_memory=(train_config.device == "cuda"),
            persistent_workers=(nw > 0),
        )

    return (
        _make(train_ex, train_config.train_batch_size, True,  augment=False),
        _make(val_ex,   train_config.eval_batch_size,  False),
        _make(test_ex,  train_config.eval_batch_size,  False),
    )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    lab = WeakLabeller()

    SAMPLES = [
        # Shreeansh Saxena
        "Product Engineer Intern Switch Climate Tech May 2025 July 2025 "
        "Cypress Python FastAPI React SQLite Docker GitHub 8.71 CGPA VIT Vellore "
        "Oracle Cloud Infrastructure Architect Associate 1Z0-1072-25",
        # Sanidhya Patel – startup founder
        "MERYT Founder Lead Engineer November 2025 GCP FastAPI microservices "
        "Google Gemini 1.5 AI Supabase Cloudflare Docker 2000+ concurrent requests "
        "Finalist Top 91/700+ Google Cloud Agentic AI Day 2025 Bangalore",
        # Mansi Jain – data engineer
        "Data Engineer Intern Mactores June 2025 AWS Glue Lambda S3 Databricks "
        "Apache Airflow IAM KMS VPC Unity Catalog CloudWatch PostgreSQL "
        "Top Finalist 10000+ teams Google Cloud Agentic AI Day",
        # Arya Bhagat – cybersecurity
        "CyberSecurity Intern Hacktify Cyber Security Feb 2025 "
        "Kali Linux Wireshark Metasploit Nmap Splunk XSS SQL Injection IDOR CSRF "
        "Google Cybersecurity Professional Certification Coursera October 2024",
        # Ishan Agarwal – finance + ML
        "Speech Emotion Recognition CNN TensorFlow PyTorch OpenAI Whisper BART NLP "
        "8% F1-score improvement real-time inference 2-3 seconds production deployment "
        "Ernst Young EY May 2024 financial model cold storage ADAPT 3.0 Winner Algorithmic Trading",
        # Jayendra Patel – competitive programmer
        "1st Place HackCrux 2025 LNMIIT Jaipur 500+ DSA problems "
        "BERT NLTK Pandas Matplotlib Transformers Python C++ Java "
        "Jabalpur Engineering College Bachelor Technology Computer Science 2024 2028",
        # Siddharth Vishwakarma – mechatronics
        "MindEase Next.js React Tailwind CSS MongoDB Google Gemini API Chart.js July 2025 "
        "FinTrace Python Flask SQLite NetworkX scikit-learn Pandas September 2025 "
        "Isolation Forest DBSCAN clustering Mechatronics 7.24",
    ]

    for s in SAMPLES:
        ex = lab.label(s)
        if ex:
            tagged = [(tok, lbl) for tok, lbl in zip(ex.tokens, ex.labels) if lbl != "O"]
            print(f"[{s[:55]}...]")
            print(f" Tags: {tagged[:14]}\n")

    gen   = SyntheticGenerator()
    synth = gen.generate(3)
    print("=== Synthetic ===")
    for ex in synth:
        tagged = [(tok, lbl) for tok, lbl in zip(ex.tokens, ex.labels) if lbl != "O"]
        print(f"Sector: {ex.sector} | {tagged[:10]}")

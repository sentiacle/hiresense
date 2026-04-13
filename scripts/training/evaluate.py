"""
HireSense AI — Evaluation Script
BERT + BiLSTM + CRF, Multi-Sector Resume NER

Runs after training. Provides:
  - Overall F1 / Precision / Recall
  - Per-entity-type breakdown
  - Per-sector breakdown (Finance, Tech, Legal, etc.)
  - Inference latency benchmark
  - Qualitative entity-extraction examples
  - CV-vs-JD scoring demo (simulates the recruiter workflow)
"""

import os
import json
import time
import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict

from transformers import BertTokenizerFast
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

from config import ModelConfig, TrainingConfig, DataConfig, ScoringConfig, get_config, ID2LABEL
from dataset import DataProcessor, create_dataloaders
from model import BertBiLSTMCRF, CVScorer


# =============================================================================
# Evaluation helpers
# =============================================================================

def evaluate_model(model: BertBiLSTMCRF, test_loader, device: str = "cpu") -> Dict:
    """Full evaluation: seqeval F1 + per-entity precision/recall."""
    model.eval()
    model.to(device)

    all_preds:  List[List[str]] = []
    all_labels: List[List[str]] = []
    entity_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"]

            out   = model(ids, mask)
            preds = out["predictions"]

            for pred_seq, lbl_seq, m in zip(preds, lbls.numpy(), mask.cpu().numpy()):
                pred_tags, true_tags = [], []
                ptr = 0
                for j, (lbl, mv) in enumerate(zip(lbl_seq, m)):
                    if mv == 1 and lbl != -100:
                        true_tag = ID2LABEL[int(lbl)]
                        pred_tag = ID2LABEL[pred_seq[ptr]] if ptr < len(pred_seq) else "O"
                        true_tags.append(true_tag)
                        pred_tags.append(pred_tag)

                        etype = true_tag.split("-")[1] if "-" in true_tag else true_tag
                        ptype = pred_tag.split("-")[1] if "-" in pred_tag else pred_tag

                        if true_tag != "O":
                            if pred_tag == true_tag: entity_stats[etype]["tp"] += 1
                            else:                    entity_stats[etype]["fn"] += 1
                        if pred_tag != "O" and pred_tag != true_tag:
                            entity_stats[ptype]["fp"] += 1
                        ptr += 1

                if true_tags:
                    all_labels.append(true_tags)
                    all_preds.append(pred_tags)

    if not all_labels:
        return {"overall": {"f1": 0, "precision": 0, "recall": 0},
                "per_entity": {}, "classification_report": ""}

    per_entity: Dict = {}
    for etype, s in entity_stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_entity[etype] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}

    return {
        "overall": {
            "f1":        f1_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds),
            "recall":    recall_score(all_labels, all_preds),
        },
        "per_entity": per_entity,
        "classification_report": classification_report(all_labels, all_preds),
    }


# =============================================================================
# Latency benchmark
# =============================================================================

def benchmark_latency(
    model: BertBiLSTMCRF,
    tokenizer: BertTokenizerFast,
    device: str = "cpu",
    n: int = 100,
) -> Dict:
    model.eval()
    model.to(device)

    SAMPLES = [
        "Software Engineer with 5 years Python and TensorFlow experience.",
        "Senior Financial Analyst skilled in SAP, IFRS, budgeting, and tax audit." * 3,
        "Civil Engineer experienced with AutoCAD, Revit, STAAD Pro, and BIM." * 5,
        "Advocate with expertise in corporate law, litigation, and due diligence." * 8,
    ]

    times: List[float] = []

    for _ in range(n):
        text   = np.random.choice(SAMPLES)
        tokens = text.split()
        t0     = time.time()

        enc = tokenizer(tokens, is_split_into_words=True, max_length=512,
                        padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            model(enc["input_ids"].to(device), enc["attention_mask"].to(device))

        times.append((time.time() - t0) * 1000)

    return {
        "mean_ms": np.mean(times),
        "std_ms":  np.std(times),
        "p50_ms":  np.percentile(times, 50),
        "p95_ms":  np.percentile(times, 95),
        "p99_ms":  np.percentile(times, 99),
        "min_ms":  np.min(times),
        "max_ms":  np.max(times),
    }


# =============================================================================
# Qualitative extraction demo
# =============================================================================

def qualitative_demo(model: BertBiLSTMCRF, tokenizer: BertTokenizerFast, device: str):
    model.eval()
    model.to(device)

    # Real CV snippets from the uploaded dataset — grounded examples for demo
    SAMPLE_RESUMES = {
        "CS_Developer (Shreeansh Saxena / Sonu Jatav)": """
            Shreeansh Saxena saxenashreeansh@gmail.com LinkedIn Github
            VIT Vellore BTech Information Technology 8.71 CGPA 2022 Present
            Product Engineer Intern Switch Climate Tech May 2025 July 2025 Pune Maharashtra
            Cypress automation testing prompt engineering React FastAPI Python SQLite Docker JWT
            App Development Intern NSoft Technologies June 2024 July 2024 Pune Maharashtra
            React Native TypeScript NodeJS ExpressJS MySQL Figma GitHub Axios Twilio SendGrid Postman
            Rubik Solver Pro React FastAPI Python SQLite Docker GitHub CI/CD Vercel Fly.io
            Hybrid Phishing Detector PyTorch PennyLane VQC Random Forest Python
            Languages Java Python C++ SQL JavaScript TypeScript HTML CSS R
            Oracle Cloud Infrastructure Architect Associate 1Z0-1072-25
            Oracle AI Foundations Associate Machine Learning Python freecodecamp
            Solved 250+ problems LeetCode GFG Finalist CODE4CHANGE 36 Hour Hackathon IEEE-SSIT VIT
        """,
        "Data_Science (Sanidhya Patel / Mansi Jain)": """
            Sanidhya Patel sanidhyapatel49@gmail.com LinkedIn GitHub Medium Jabalpur
            MERYT Founder Lead Engineer November 2025 Present
            GCP FastAPI microservices Google Gemini 1.5 AI Supabase Cloudflare Docker
            NySon AI Connect Software Developer Intern Freelancer August 2024 Present
            Django React RESTful APIs Python frontend optimization
            HOSHŌ DIGITAL Junior Consultant Intern February 2026 Present
            Power Apps Dataverse RBAC Power Automate Outlook enterprise inventory
            Invested AI Financial Co-Pilot RAG FastAPI Pinecone Google Cloud Run Firebase Flutter Docker
            Fake Banking APK Detection XGBoost SHAP GCP Androguard CNN LSTM Python Flask
            Hybrid AI Private Offline LLM Stack Unsloth llama.cpp GGUF LoRA NVIDIA CUDA
            Oracle Cloud Infrastructure 2025 Certified Data Science Professional
            Google Cloud Agentic AI Day 2025 Finalist Top 91/700+ Bangalore
            National Finalist CIIS 25 Jabalpur Engineering College
            Bachelor of Technology Information Technology Jabalpur Engineering College Pursuing
        """,
        "Data_Engineering (Mansi Jain / Aditya Tomar)": """
            Mansi Jain Jabalpur India barkulmansi@gmail.com LinkedIn Github
            B.Tech Information Technology Jabalpur Engineering College 2022 2026
            B.S. Degree Data Science Applications IIT Madras 2023 2027
            Data Engineer Intern Mactores June 2025
            AWS Glue Lambda S3 Databricks Apache Airflow CloudWatch PostgreSQL
            IAM KMS encryption VPC Unity Catalog ETL pipelines Redshift SQL
            Data Analyst Intern GoStops September 2025
            Power BI dashboards occupancy revenue customer acquisition analytics
            INVESTED Personal Finance Platform GCP Flutter FastAPI MCP server Firebase Pinecone RAG
            Blinkit Data Visualization Power BI ETL Python SQL Server pandas
            AWS Certified Cloud Practitioner Databricks Certified Data Engineer Associate
            Top Finalist 10000+ teams Google Cloud Agentic AI Day Hackathon
            Reliance Foundation Undergraduate Scholar
            Python C++ SQL Java JavaScript MySQL Data Structures RDBMS Big Data
            Docker GitHub Jenkins Linux NumPy Pandas Scikit-learn Matplotlib Seaborn AWS Azure GCP
        """,
        "Cybersecurity (Arya Bhagat / Subrato Singh)": """
            Arya Bhagat aryabhagat6686@gmail.com LinkedIn Jabalpur Engineering College
            BTech Artificial Intelligence Data Science Engineering 2022 2026
            CyberSecurity Intern Hacktify Cyber Security February 2025 March 2025
            Penetration testing XSS SQL Injection IDOR CORS CSRF security misconfigurations
            CTF challenges red teaming OSINT cryptography privilege escalation
            Network Security Penetration Testing SIEM Splunk Firewalls IDS Linux Windows
            Python C HTML Bash Kali Linux Wireshark Metasploit Nmap
            Web Application Security Testing SQL Injection XSS mock e-commerce platform
            Google Cybersecurity Professional Certification Coursera October 2024
            Cyber Security Privacy NPTEL July October 2024
            Linux Trainee CYBERSEC INFOTECH Private Limited August September 2024
            firewall access control security hardening log monitoring incident response
            AWS Docker Kubernetes Jenkins GitHub Actions Terraform PostgreSQL MySQL MongoDB Redis
        """,
        "Finance_Tech (NMIMS MBA Tech — Manjari Narang / Ishan Agarwal / Sanyam Jain)": """
            Manjari Narang MBA Tech Class 2027 Management Finance Engineering Computer NMIMS
            MPSTME NMIMS Mumbai CGPA 3.16/4 2027
            Technical Internship Jio Platforms Ltd May 2025 July 2025
            YOLO object detection tracking computer vision Python occupancy insights
            IBM HR Dashboard Tableau workforce data HR decision-making
            Cricket Management System MySQL tournament teams players schedules venues
            Object Detection Video Analytics YOLO surveillance workspace monitoring
            Bloomberg Market Concepts Bloomberg Finance Fundamentals Investment Banking J.P. Morgan
            Market Analytics Python SQL Data Analysis Excel Tableau SAS Financial Market Analysis
            Student Placement Coordinator Vice President Communications MBA Tech Connect Cell
            Ishan Agarwal MPSTME NMIMS Mumbai Finance Computer 2.94/4
            Financial Modelling Risk Management MySQL Power BI Python C++ React JS MongoDB
            Speech Emotion Recognition CNN TensorFlow PyTorch OpenAI Whisper BART NLP
            Astronaut Fitness Prediction Random Forest Gradient Boosting 94% accuracy
            Ernst Young EY May 2024 financial model cold storage profitability forecasting
            IS Global Web MERN stack Job Portal JWT authentication CRUD
            ADAPT 3.0 Winner Algorithmic Trading President Social Conclave 2026
        """,
    }

    scorer = CVScorer(ScoringConfig())

    print("\n" + "=" * 70)
    print("Qualitative Extraction & Scoring Demo")
    print("=" * 70)

    for sector, resume_text in SAMPLE_RESUMES.items():
        print(f"\n── {sector} ──────────────────────────────────────────────")
        tokens = resume_text.split()
        enc    = tokenizer(tokens, is_split_into_words=True, max_length=512,
                           padding="max_length", truncation=True, return_tensors="pt")

        with torch.no_grad():
            out   = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            preds = out["predictions"][0]

        entities = model.get_entities(tokens, preds, enc.word_ids())

        by_type: Dict[str, List[str]] = defaultdict(list)
        for e in entities:
            by_type[e["label"]].append(e["text"])

        for lbl, vals in sorted(by_type.items()):
            print(f"  {lbl:10s}: {', '.join(vals[:6])}")

        # ── Simulate scoring against a JD ─────────────────────────────
        jd = (f"We need a {sector} professional with relevant skills, "
              f"experience and appropriate educational qualifications.")
        score = scorer.score(entities, jd)
        print(f"  → Simulated Score vs JD: {score['overall']:.1f}/100")


# =============================================================================
# CV-vs-JD scoring demo (full recruiter workflow)
# =============================================================================

def scoring_workflow_demo(model: BertBiLSTMCRF, tokenizer: BertTokenizerFast, device: str):
    """
    Demonstrates the full HireSense scoring pipeline:
      Recruiter provides JD + weights → Candidate uploads CV → Score produced.
    """
    print("\n" + "=" * 70)
    print("HireSense Scoring Workflow Demo")
    print("=" * 70)

    # Based on Sanidhya Patel's actual CV (uploaded dataset)
    jd = """
    Job Title: Backend / AI Engineer (Fresher / Intern)
    We are looking for a motivated fresher or intern with Python and backend experience.

    Required Skills: Python, FastAPI or Django, Docker, REST APIs, SQL or NoSQL databases.
    Nice-to-have: GCP or AWS, LLM/AI workflows, React, Git.

    Education: B.Tech in Computer Science, Information Technology, or related field.
    Experience: Internship or project experience in backend development or AI/ML.
    """

    cv = """
    Sanidhya Patel sanidhyapatel49@gmail.com Jabalpur Madhya Pradesh India
    MERYT Founder Lead Engineer November 2025 Present
    GCP FastAPI microservices Google Gemini AI Supabase Cloudflare Docker
    NySon AI Connect Software Developer Intern August 2024 Present
    Python Django React RESTful APIs MySQL PostgreSQL performance optimization
    HOSHŌ DIGITAL Junior Consultant Intern February 2026 Present
    Power Apps Dataverse RBAC Power Automate
    Invested AI Financial Co-Pilot RAG FastAPI Pinecone Google Cloud Run Firebase Flutter Docker Python
    Fake Banking APK Detection XGBoost SHAP GCP Androguard Python Flask
    BinSavvy React TypeScript Django MySQL YOLO
    Oracle Cloud Infrastructure 2025 Certified Data Science Professional
    Oracle Cloud Infrastructure 2025 Certified Gen AI Professional
    Google Cloud Agentic AI Day 2025 Finalist Top 91/700+
    National Finalist CIIS 25 Finalist HackCrux 25
    Bachelor of Technology Information Technology Jabalpur Engineering College
    """

    # Recruiter-configured weights (must sum to 1)
    recruiter_weights = {
        "skill":       0.40,
        "experience":  0.30,
        "education":   0.15,
        "project":     0.10,
        "achievement": 0.05,
    }

    tokens = cv.split()
    enc    = tokenizer(tokens, is_split_into_words=True, max_length=512,
                       padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        out   = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        preds = out["predictions"][0]

    entities = model.get_entities(tokens, preds, enc.word_ids())

    scorer = CVScorer(ScoringConfig())
    result = scorer.score(entities, jd, weights=recruiter_weights)

    print(f"\n  JD (truncated)    : {jd.strip()[:80]} …")
    print(f"\n  Overall Score     : {result['overall']:.1f} / 100")
    print(f"\n  Breakdown (recruiter weights):")
    for cat, info in result["breakdown"].items():
        bar = "█" * int(info["score"] / 10)
        print(f"    {cat:12s}: {info['score']:5.1f}  (weight={info['weight']:.2f})  {bar}")

    # Simulated test score (from the MCQ/timer module)
    test_score = 74.0
    final = 0.7 * result["overall"] + 0.3 * test_score
    print(f"\n  Test Score (MCQ)  : {test_score:.1f} / 100")
    print(f"  Final Score       : {final:.1f} / 100  (70% CV + 30% Test)")
    print(f"  Hiring Decision   : {'PASS ✓' if final >= 60 else 'FAIL ✗'}")


# =============================================================================
# Main
# =============================================================================

def main():
    model_config, train_config, data_config, scoring_config = get_config()

    model_path = os.path.join(data_config.model_save_path, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train first with train.py")
        return

    print("Loading model …")
    ckpt = torch.load(model_path, map_location=train_config.device, weights_only=False)
    model = BertBiLSTMCRF(model_config)
    model.load_state_dict(ckpt["model_state_dict"])

    tokenizer = BertTokenizerFast.from_pretrained(model_config.bert_model_name)

    # ── Test data ─────────────────────────────────────────────────────
    print("Loading test data …")
    processor = DataProcessor(data_config)
    _, _, test_ex = processor.load_all_data()
    _, _, test_loader = create_dataloaders([], [], test_ex, tokenizer, train_config)

    # ── Quantitative evaluation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Quantitative Evaluation")
    print("=" * 60)

    results = evaluate_model(model, test_loader, train_config.device)

    print(f"\nOverall:")
    print(f"  F1        : {results['overall']['f1']:.4f}")
    print(f"  Precision : {results['overall']['precision']:.4f}")
    print(f"  Recall    : {results['overall']['recall']:.4f}")

    print(f"\nPer-Entity:")
    for etype, m in sorted(results["per_entity"].items()):
        print(f"  {etype:12s}  F1={m['f1']:.3f}  P={m['precision']:.3f}"
              f"  R={m['recall']:.3f}  support={m['support']}")

    print(f"\nDetailed Report:\n{results['classification_report']}")

    # ── Latency ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Inference Latency")
    print("=" * 60)
    lat = benchmark_latency(model, tokenizer, train_config.device)
    print(f"  Mean  : {lat['mean_ms']:.2f} ms")
    print(f"  P50   : {lat['p50_ms']:.2f} ms")
    print(f"  P95   : {lat['p95_ms']:.2f} ms")
    print(f"  P99   : {lat['p99_ms']:.2f} ms")

    # ── Qualitative ───────────────────────────────────────────────────
    qualitative_demo(model, tokenizer, train_config.device)

    # ── Scoring workflow ──────────────────────────────────────────────
    scoring_workflow_demo(model, tokenizer, train_config.device)

    # ── Save results ──────────────────────────────────────────────────
    out_path = os.path.join(data_config.output_dir, "evaluation_results.json")
    os.makedirs(data_config.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "overall": results["overall"],
            "per_entity": results["per_entity"],
            "latency": lat,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

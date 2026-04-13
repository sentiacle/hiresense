"""
HireSense AI — Training Configuration
BERT + BiLSTM + CRF, Multi-Sector Resume NER

Changes for higher scoring accuracy:
  - ScoringConfig.default_weights tuned from real CV analysis
  - SYNONYMS moved here so both model.py and dataset.py share them
  - Added SECTION_HEADERS: common resume section names that should
    never be scored as entities (prevents false positives)
"""

from dataclasses import dataclass, field
from typing import List, Dict
import torch


@dataclass
class ModelConfig:
    bert_model_name: str = "bert-base-uncased"
    bert_hidden_size: int = 768
    freeze_bert: bool = False
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True

    @property
    def lstm_output_size(self) -> int:
        return self.lstm_hidden_size * (2 if self.lstm_bidirectional else 1)


@dataclass
class TrainingConfig:
    max_seq_length: int = 512
    train_batch_size: int = 8
    eval_batch_size: int = 16

    bert_learning_rate: float = 2e-5
    lstm_crf_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    warmup_ratio: float = 0.1
    num_epochs: int = 15
    gradient_accumulation_steps: int = 4

    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = torch.cuda.is_available()
    seed: int = 42


@dataclass
class DataConfig:
    kaggle_pdf_path: str = "./data/kaggle_resume_pdf"
    resume_corpus_path: str = "./data/resume_corpus"
    ner_annotated_path: str = "./data/ner_annotated_cvs"
    output_dir: str = "./output"
    model_save_path: str = "./output"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class ScoringConfig:
    """
    Recruiter-configurable weights for CV-vs-JD scoring.
    Defaults tuned from analysis of the 20 real uploaded CVs:
      - Skills are the single biggest differentiator for tech/finance roles
      - Experience matters almost as much
      - Projects are a strong signal for freshers (no full-time exp yet)
      - Education and achievements are tie-breakers
    """
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        "skill":       0.35,
        "experience":  0.25,
        "education":   0.15,
        "project":     0.15,   # raised from 0.10 — critical for freshers
        "achievement": 0.10,
    })
    min_entity_threshold: int = 1
    similarity_threshold: float = 0.70   # lowered slightly for better recall


# =============================================================================
# Shared skill/term synonyms used by both scorer and labeller
# =============================================================================

SKILL_SYNONYMS: Dict[str, List[str]] = {
    # Languages
    "python":         ["py"],
    "javascript":     ["js", "ecmascript", "es6", "es2015"],
    "typescript":     ["ts"],
    "c++":            ["cpp", "c plus plus"],
    "c#":             ["csharp", "c sharp"],
    # Frameworks
    "react":          ["reactjs", "react.js"],
    "node":           ["nodejs", "node.js"],
    "next":           ["nextjs", "next.js"],
    "express":        ["expressjs", "express.js"],
    "django":         ["django rest framework", "drf"],
    "fastapi":        ["fast api"],
    "flask":          ["micro flask"],
    # Databases
    "postgresql":     ["postgres", "psql"],
    "mongodb":        ["mongo", "nosql"],
    "mysql":          ["sql server"],
    "sqlite":         ["sqlite3"],
    # Cloud / DevOps
    "aws":            ["amazon web services", "amazon aws"],
    "gcp":            ["google cloud", "google cloud platform"],
    "azure":          ["microsoft azure"],
    "docker":         ["containerization", "containers"],
    "kubernetes":     ["k8s"],
    "ci/cd":          ["continuous integration", "github actions", "jenkins", "devops"],
    "git":            ["github", "gitlab", "version control"],
    # ML / AI
    "machine learning": ["ml", "ai", "artificial intelligence"],
    "deep learning":  ["dl", "neural networks", "ann"],
    "nlp":            ["natural language processing", "text processing"],
    "computer vision":["cv", "image processing", "object detection"],
    "pytorch":        ["torch"],
    "tensorflow":     ["tf", "keras"],
    "scikit-learn":   ["sklearn"],
    "rag":            ["retrieval augmented generation", "vector search"],
    "llm":            ["large language model", "gpt", "gemini", "llama"],
    # Finance
    "financial modelling": ["financial modeling", "financial model", "financial analysis",
                             "financial reporting"],
    "bloomberg":      ["bloomberg terminal", "bloomberg finance", "bloomberg market concepts"],
    "excel":          ["microsoft excel", "ms excel", "spreadsheet"],
    "power bi":       ["powerbi", "bi dashboard"],
    "tableau":        ["data viz", "data visualization"],
    "sql":            ["structured query language", "rdbms"],
    "accounting":     ["accounts", "bookkeeping", "tally", "gst"],
    "risk management":["risk analysis", "risk assessment"],
    # HR
    "human resources":["hr", "people ops", "talent acquisition", "recruitment"],
    # Security
    "penetration testing": ["pentest", "pen testing", "ethical hacking"],
    "kali linux":     ["kali"],
    "wireshark":      ["packet analysis", "network analysis"],
    # Data Engineering
    "apache airflow": ["airflow", "workflow orchestration"],
    "databricks":     ["spark", "lakehouse"],
    "etl":            ["data pipeline", "data engineering"],
    "aws glue":       ["glue", "data catalog"],
    # General
    "rest api":       ["restful", "restful api", "api development"],
    "microservices":  ["service mesh", "distributed systems"],
    "agile":          ["scrum", "kanban", "sprint"],
    "project management": ["pmp", "prince2"],
}

# =============================================================================
# Entity label schema (BIO)
# =============================================================================

ENTITY_LABELS: List[str] = [
    "O",
    "B-SKILL",   "I-SKILL",
    "B-EXP",     "I-EXP",
    "B-EDU",     "I-EDU",
    "B-PROJ",    "I-PROJ",
    "B-ACH",     "I-ACH",
    "B-CERT",    "I-CERT",
    "B-ORG",     "I-ORG",
    "B-LOC",     "I-LOC",
    "B-DATE",    "I-DATE",
    "B-NAME",    "I-NAME",
    "B-CONTACT", "I-CONTACT",
    "B-SECTOR",  "I-SECTOR",
]

LABEL2ID  = {label: idx for idx, label in enumerate(ENTITY_LABELS)}
ID2LABEL  = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(ENTITY_LABELS)

ENTITY_GROUPS: Dict[str, List[str]] = {
    "skill":        ["B-SKILL",   "I-SKILL"],
    "experience":   ["B-EXP",     "I-EXP"],
    "education":    ["B-EDU",     "I-EDU"],
    "project":      ["B-PROJ",    "I-PROJ"],
    "achievement":  ["B-ACH",     "I-ACH",  "B-CERT", "I-CERT"],
    "organization": ["B-ORG",     "I-ORG"],
    "date":         ["B-DATE",    "I-DATE"],
    "sector":       ["B-SECTOR",  "I-SECTOR"],
}

# Section headers that appear in resumes but should NEVER be scored as entities
# (prevents false positives from section titles being tagged as skills/exp)
RESUME_SECTION_HEADERS = {
    "education", "experience", "skills", "projects", "achievements",
    "certifications", "awards", "publications", "summary", "objective",
    "profile", "contact", "work experience", "technical skills",
    "core skills", "internships", "positions of responsibility",
    "extracurricular", "activities", "interests", "languages",
    "startup", "startups", "academic projects", "competitions",
}


def get_config():
    return ModelConfig(), TrainingConfig(), DataConfig(), ScoringConfig()


if __name__ == "__main__":
    mc, tc, dc, sc = get_config()
    print(f"Model   : {mc.bert_model_name} + BiLSTM({mc.lstm_hidden_size}) + CRF")
    print(f"Labels  : {NUM_LABELS}")
    print(f"Device  : {tc.device}")
    print(f"Weights : {sc.default_weights}")
    print(f"Synonyms: {len(SKILL_SYNONYMS)} entries")

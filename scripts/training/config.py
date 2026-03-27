"""
HireSense AI - Training Configuration
BERT + BiLSTM + CRF for Resume NER
"""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # BERT configuration
    bert_model_name: str = "bert-base-uncased"
    bert_hidden_size: int = 768
    freeze_bert: bool = False  # Whether to freeze BERT layers
    
    # BiLSTM configuration
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True
    
    # Output size is 2 * lstm_hidden_size if bidirectional
    @property
    def lstm_output_size(self) -> int:
        return self.lstm_hidden_size * (2 if self.lstm_bidirectional else 1)


@dataclass
class TrainingConfig:
    """
    Training hyperparameters
    Optimized for modern GPUs (e.g., RTX 4060 Ti)
    """
    # Data - batch_size=8 fits comfortably in 8GB VRAM with BERT
    max_seq_length: int = 512
    train_batch_size: int = 8  # Adjust based on VRAM, use gradient accumulation
    eval_batch_size: int = 16
    
    # Optimizer
    bert_learning_rate: float = 2e-5
    lstm_crf_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Scheduler
    warmup_ratio: float = 0.1
    num_epochs: int = 15  # More epochs since we have smaller batches
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Early stopping
    early_stopping_patience: int = 5  # More patience for NER
    early_stopping_threshold: float = 0.001
    
    # Device - Auto-detect CUDA
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = torch.cuda.is_available()  # Enable FP16 for modern GPUs
    
    # Reproducibility
    seed: int = 42
    
    # Gradient accumulation (effective batch = train_batch_size * accumulation_steps)
    gradient_accumulation_steps: int = 4  # Effective batch of 32


@dataclass
class DataConfig:
    """Dataset configuration"""
    # Paths to different datasets
    # The main one from Kaggle for weak supervision
    kaggle_pdf_path: str = "./data/kaggle_resume_pdf"
    resume_corpus_path: str = "./data/resume_corpus"
    ner_annotated_path: str = "./data/ner_annotated_cvs"
    
    # Output paths
    output_dir: str = "./output"
    model_save_path: str = "./output/model"
    
    # Train/val/test split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


# Entity labels for Resume NER (BIO format)
ENTITY_LABELS: List[str] = [
    "O",           # Outside any entity
    "B-SKILL",     # Beginning of skill
    "I-SKILL",     # Inside skill
    "B-EXP",       # Beginning of experience
    "I-EXP",       # Inside experience
    "B-EDU",       # Beginning of education
    "I-EDU",       # Inside education
    "B-PROJ",      # Beginning of project
    "I-PROJ",      # Inside project
    "B-ACH",       # Beginning of achievement
    "I-ACH",       # Inside achievement
    "B-ORG",       # Beginning of organization
    "I-ORG",       # Inside organization
    "B-LOC",       # Beginning of location
    "I-LOC",       # Inside location
    "B-DATE",      # Beginning of date
    "I-DATE",      # Inside date
    "B-NAME",      # Beginning of person name
    "I-NAME",      # Inside person name
    "B-CONTACT",   # Beginning of contact info
    "I-CONTACT",   # Inside contact info
    "B-CERT",      # Beginning of certification
    "I-CERT",      # Inside certification
]

# Create label to index mapping
LABEL2ID = {label: idx for idx, label in enumerate(ENTITY_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(ENTITY_LABELS)

# Entity type groupings for scoring
ENTITY_GROUPS = {
    "skills": ["B-SKILL", "I-SKILL"],
    "experience": ["B-EXP", "I-EXP"],
    "education": ["B-EDU", "I-EDU"],
    "projects": ["B-PROJ", "I-PROJ"],
    "achievements": ["B-ACH", "I-ACH", "B-CERT", "I-CERT"],
    "organizations": ["B-ORG", "I-ORG"],
    "dates": ["B-DATE", "I-DATE"],
}


def get_config():
    """Get all configuration objects"""
    return ModelConfig(), TrainingConfig(), DataConfig()


if __name__ == "__main__":
    model_cfg, train_cfg, data_cfg = get_config()
    print(f"Model: {model_cfg.bert_model_name}")
    print(f"LSTM hidden: {model_cfg.lstm_hidden_size}")
    print(f"Num labels: {NUM_LABELS}")
    print(f"Device: {train_cfg.device}")
    print(f"Labels: {ENTITY_LABELS}")

"""
HireSense AI - Model Loading and Caching
BERT + BiLSTM + CRF for Resume NER
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from functools import lru_cache

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from torchcrf import CRF


# Entity labels
ENTITY_LABELS = [
    "O", "B-SKILL", "I-SKILL", "B-EXP", "I-EXP", "B-EDU", "I-EDU",
    "B-PROJ", "I-PROJ", "B-ACH", "I-ACH", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-DATE", "I-DATE", "B-NAME", "I-NAME",
    "B-CONTACT", "I-CONTACT", "B-CERT", "I-CERT", "B-SECTOR", "I-SECTOR"
]

LABEL2ID = {label: idx for idx, label in enumerate(ENTITY_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(ENTITY_LABELS)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
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


class BertBiLSTMCRF(nn.Module):
    """BERT + BiLSTM + CRF for Named Entity Recognition"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_labels = NUM_LABELS
        
        # BERT
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        
        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=config.bert_hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(config.lstm_dropout)
        self.hidden2tag = nn.Linear(config.lstm_output_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        emissions = self.hidden2tag(lstm_output)
        
        outputs = {"emissions": emissions}
        
        if labels is not None:
            mask = (labels != -100) & (attention_mask == 1)
            labels_for_crf = labels.clone()
            labels_for_crf[labels == -100] = 0
            loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction='mean')
            outputs["loss"] = loss
        
        # Viterbi decoding
        with torch.no_grad():
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            outputs["predictions"] = predictions
            
        return outputs
    
    def get_entities(
        self,
        tokens: List[str],
        predictions: List[int],
        word_ids: List[Optional[int]]
    ) -> List[Dict]:
        """Extract entities from predictions"""
        entities = []
        current_entity = None
        
        # Map predictions back to words
        word_preds = []
        prev_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_idx:
                if idx < len(predictions):
                    word_preds.append((word_idx, predictions[idx]))
            prev_idx = word_idx
        
        # Extract using BIO scheme
        for word_idx, pred_id in word_preds:
            if word_idx >= len(tokens):
                continue
                
            label = ID2LABEL.get(pred_id, "O")
            token = tokens[word_idx]
            
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = {
                    "text": token,
                    "label": entity_type,
                    "start": word_idx,
                    "end": word_idx,
                    "confidence": 1.0
                }
            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    current_entity["text"] += " " + token
                    current_entity["end"] = word_idx
                else:
                    entities.append(current_entity)
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
            
        return entities


class ModelManager:
    """Singleton model manager for loading and caching models"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.model: Optional[BertBiLSTMCRF] = None
        self.tokenizer: Optional[BertTokenizerFast] = None
        self.config: Optional[ModelConfig] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.model_path = os.environ.get("MODEL_PATH", "./models/resume_ner_model.pt")
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load model from checkpoint"""
        if model_path:
            self.model_path = model_path
            
        try:
            print(f"Loading model from {self.model_path}...")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Model file not found at {self.model_path}")
                print("Using fallback heuristic model...")
                self._init_fallback_model()
                return True
            
            # Fix for "No module named 'config'" when PyTorch unpickles the model
            import sys
            if 'config' not in sys.modules:
                sys.modules['config'] = sys.modules[__name__]

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Get config
            if "config" in checkpoint:
                self.config = checkpoint["config"]
            else:
                self.config = ModelConfig()
            
            # Initialize model
            self.model = BertBiLSTMCRF(self.config)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            self.tokenizer = BertTokenizerFast.from_pretrained(
                self.config.bert_model_name
            )
            
            self.model_loaded = True
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback heuristic model...")
            self._init_fallback_model()
            return True
    
    def _init_fallback_model(self):
        """Initialize fallback when trained model not available"""
        self.config = ModelConfig()
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = None  # Will use heuristic extraction
        self.model_loaded = False
        
    def extract_entities(self, text: str) -> Tuple[List[Dict], List[str], List[str]]:
        """
        Extract entities from text
        Returns: (entities, tokens, labels)
        """
        import re
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        if self.model is not None and self.model_loaded:
            # Use trained model
            return self._extract_with_model(tokens)
        else:
            # Use heuristic fallback
            return self._extract_with_heuristics(tokens, text)
    
    def _extract_with_model(self, tokens: List[str]) -> Tuple[List[Dict], List[str], List[str]]:
        """Extract entities using trained BERT model"""
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        word_ids = encoding.word_ids()
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            predictions = outputs["predictions"][0]
        
        entities = self.model.get_entities(tokens, predictions, word_ids)
        
        # Get labels for each token
        labels = []
        pred_idx = 0
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:
                if pred_idx < len(predictions):
                    labels.append(ID2LABEL.get(predictions[pred_idx], "O"))
                    pred_idx += 1
            prev_word_idx = word_idx
        
        return entities, tokens, labels
    
    def _extract_with_heuristics(self, tokens: List[str], text: str) -> Tuple[List[Dict], List[str], List[str]]:
        """Fallback heuristic extraction when no trained model"""
        import re
        
        entities = []
        labels = ["O"] * len(tokens)
        text_lower = text.lower()
        
        # Skill keywords
        skills = [
            "python", "javascript", "java", "c++", "c#", "ruby", "go", "rust", "swift",
            "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
            "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
            "sql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "git", "linux", "rest", "graphql", "api", "microservices",
            "machine learning", "deep learning", "nlp", "computer vision",
            "typescript", "html", "css", "sass", "webpack"
        ]
        
        for skill in skills:
            if skill.lower() in text_lower:
                # Find position
                pattern = re.compile(re.escape(skill), re.IGNORECASE)
                for match in pattern.finditer(text):
                    start_char = match.start()
                    # Map to token index
                    char_count = 0
                    for i, token in enumerate(tokens):
                        if char_count >= start_char:
                            entities.append({
                                "text": match.group(),
                                "label": "SKILL",
                                "start": i,
                                "end": i,
                                "confidence": 0.8
                            })
                            labels[i] = "B-SKILL"
                            break
                        char_count += len(token) + 1
        
        # Experience patterns
        exp_patterns = [
            r"(\d+)\+?\s*years?\s*(of\s+)?experience",
            r"(senior|lead|principal|junior|staff)\s+(software\s+)?(engineer|developer|scientist|manager)",
        ]
        
        for pattern in exp_patterns:
            for match in re.finditer(pattern, text_lower):
                entities.append({
                    "text": match.group(),
                    "label": "EXP",
                    "start": 0,
                    "end": 0,
                    "confidence": 0.7
                })
        
        # Education patterns
        edu_patterns = [
            r"(bachelor|master|phd|doctorate|b\.?s\.?|m\.?s\.?|mba)",
            r"(computer science|data science|engineering|mathematics)"
        ]
        
        for pattern in edu_patterns:
            for match in re.finditer(pattern, text_lower):
                entities.append({
                    "text": match.group(),
                    "label": "EDU",
                    "start": 0,
                    "end": 0,
                    "confidence": 0.7
                })
        
        # Organization patterns (common companies)
        orgs = ["google", "microsoft", "amazon", "meta", "apple", "netflix", "uber", "airbnb"]
        for org in orgs:
            if org in text_lower:
                entities.append({
                    "text": org.title(),
                    "label": "ORG",
                    "start": 0,
                    "end": 0,
                    "confidence": 0.9
                })
        
        # Certification patterns
        cert_patterns = [
            r"(aws|azure|gcp|google)\s+(certified|certification)",
            r"(pmp|scrum|agile)\s+(certified|master)?",
        ]
        
        for pattern in cert_patterns:
            for match in re.finditer(pattern, text_lower):
                entities.append({
                    "text": match.group(),
                    "label": "CERT",
                    "start": 0,
                    "end": 0,
                    "confidence": 0.7
                })
        
        return entities, tokens, labels
    
    def get_status(self) -> Dict:
        """Get model status"""
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.config.bert_model_name if self.config else "heuristic",
            "device": self.device,
            "model_path": self.model_path
        }


# Global model manager
@lru_cache()
def get_model_manager() -> ModelManager:
    """Get or create model manager singleton"""
    manager = ModelManager()
    manager.load_model()
    return manager

"""
HireSense AI - BERT + BiLSTM + CRF Model Architecture
For Resume Named Entity Recognition
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torchcrf import CRF
from typing import Optional, Dict, Tuple

from config import ModelConfig, NUM_LABELS, LABEL2ID, ID2LABEL

class BertBiLSTMCRF(nn.Module):
    """
    BERT + BiLSTM + CRF for Named Entity Recognition
    
    Architecture:
    1. BERT-base-uncased: Contextual embeddings (768-dim)
    2. BiLSTM: Sequence modeling with bidirectional context (256*2=512-dim)
    3. CRF: Structured prediction for valid tag sequences
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_labels = NUM_LABELS
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        
        # Optionally freeze BERT layers
        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=config.bert_hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.lstm_dropout)
        
        # Linear projection to tag space
        self.hidden2tag = nn.Linear(config.lstm_output_size, self.num_labels)
        
        # CRF layer
        self.crf = CRF(self.num_labels, batch_first=True)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize LSTM and linear layer weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
                
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: NER labels [batch, seq_len], -100 for ignored tokens
            
        Returns:
            Dictionary with loss (if labels provided) and predictions
        """
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_outputs.last_hidden_state  # [batch, seq_len, 768]
        
        # Pass through BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # [batch, seq_len, 512]
        
        # Dropout
        lstm_output = self.dropout(lstm_output)
        
        # Project to tag space
        emissions = self.hidden2tag(lstm_output)  # [batch, seq_len, num_labels]
        
        outputs = {"emissions": emissions}
        
        if labels is not None:
            # Create mask for CRF (ignore -100 labels and padding)
            # CRF mask: True for valid positions, False for ignored
            mask = attention_mask.bool().clone()
            mask[:, 0] = True
            
            # Replace -100 with 0 for CRF (it will be masked anyway)
            labels_for_crf = labels.clone()
            labels_for_crf[labels == -100] = 0
            
            # CRF loss (negative log-likelihood)
            loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction='mean')
            outputs["loss"] = loss
            
        # Get predictions using Viterbi decoding
        with torch.no_grad():
            mask = attention_mask.bool().clone()
            mask[:, 0] = True
            predictions = self.crf.decode(emissions, mask=mask)
            outputs["predictions"] = predictions
            
        return outputs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        """
        Inference-only prediction
        
        Returns:
            emissions: Raw emission scores
            predictions: Decoded tag sequences
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs["emissions"], outputs["predictions"]
    
    def get_entities(
        self,
        tokens: list,
        predictions: list,
        tokenizer_word_ids: list
    ) -> list:
        """
        Extract named entities from predictions
        
        Args:
            tokens: Original word tokens
            predictions: Predicted tag IDs
            tokenizer_word_ids: Word ID mapping from tokenizer
            
        Returns:
            List of entities: [{"text": str, "label": str, "start": int, "end": int}]
        """
        entities = []
        current_entity = None
        
        # Map predictions back to original words
        word_predictions = []
        prev_word_idx = None
        
        for idx, word_idx in enumerate(tokenizer_word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:
                if len(word_predictions) < len(predictions):
                    word_predictions.append((word_idx, predictions[len(word_predictions)]))
            prev_word_idx = word_idx
        
        # Extract entities using BIO scheme
        for word_idx, pred_id in word_predictions:
            if word_idx >= len(tokens):
                continue
                
            label = ID2LABEL.get(pred_id, "O")
            token = tokens[word_idx]
            
            if label.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                # Start new entity
                entity_type = label[2:]
                current_entity = {
                    "text": token,
                    "label": entity_type,
                    "start": word_idx,
                    "end": word_idx
                }
            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    # Continue entity
                    current_entity["text"] += " " + token
                    current_entity["end"] = word_idx
                else:
                    # Entity type mismatch, save and reset
                    entities.append(current_entity)
                    current_entity = None
            else:
                # O tag
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Don't forget last entity
        if current_entity:
            entities.append(current_entity)
            
        return entities


class ModelForInference:
    """
    Wrapper for easy inference with pre-trained model
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        
        # Load config and model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = checkpoint.get("config", ModelConfig())
        self.model = BertBiLSTMCRF(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.config.bert_model_name
        )
        
    def extract_entities(self, text: str) -> list:
        """
        Extract entities from raw text
        
        Args:
            text: Input resume text
            
        Returns:
            List of entities with text, label, and positions
        """
        import re

        # 🔥 Guard 1: empty text
        if not text or len(text.strip()) == 0:
            return []

        # Tokenization
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)

        # 🔥 Guard 2: no tokens
        if len(tokens) == 0:
            return []

        # Tokenize for BERT
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

        # 🔥 Guard 3: word_ids safety
        if word_ids is None:
            return []

        # Get predictions
        _, predictions = self.model.predict(input_ids, attention_mask)

        # 🔥 Guard 4: empty predictions
        if not predictions or len(predictions[0]) == 0:
            return []

        # Extract entities
        entities = self.model.get_entities(
            tokens,
            predictions[0],  # First (and only) batch
            word_ids
        )

        return entities


if __name__ == "__main__":
    # Test model architecture
    config = ModelConfig()
    model = BertBiLSTMCRF(config)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: BERT + BiLSTM + CRF")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"BERT hidden size: {config.bert_hidden_size}")
    print(f"LSTM hidden size: {config.lstm_hidden_size}")
    print(f"LSTM output size: {config.lstm_output_size}")
    print(f"Number of labels: {NUM_LABELS}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, NUM_LABELS, (batch_size, seq_len))
    
    outputs = model(input_ids, attention_mask, labels)
    
    print(f"\nTest forward pass:")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Emissions shape: {outputs['emissions'].shape}")
    print(f"Predictions: {len(outputs['predictions'])} sequences")

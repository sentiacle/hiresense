"""
HireSense AI — Backend Inference Helper
Drop this file into your FastAPI backend alongside resume_ner_model.pt.
"""
import re, torch
from pathlib import Path
from transformers import BertTokenizerFast
from model import BertBiLSTMCRF          # copy model.py to backend too
from config import ModelConfig            # copy config.py to backend too


class ResumeNERPipeline:
    """Load once; call extract() per request."""

    def __init__(self, model_dir: str = "./models", device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(
            Path(model_dir) / "resume_ner_model.pt",
            map_location=self.device, weights_only=False,
        )
        self.model = BertBiLSTMCRF(ckpt.get("config", ModelConfig()))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        self.tokenizer = BertTokenizerFast.from_pretrained(Path(model_dir) / "tokenizer")
        self.id2label  = ckpt.get("id2label", {})

    def extract(self, text: str) -> list:
        if not text or not text.strip():
            return []
        tokens = re.findall(r"\b\w[\w'.+-]*\b|[^\w\s]", text)
        if not tokens:
            return []
        enc = self.tokenizer(
            tokens, is_split_into_words=True,
            max_length=512, padding="max_length", truncation=True, return_tensors="pt",
        )
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out   = self.model(ids, mask)
            preds = out["predictions"][0]
        return self.model.get_entities(tokens, preds, enc.word_ids())

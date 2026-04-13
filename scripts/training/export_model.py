"""
HireSense AI — Model Export Script
Packages the trained model for deployment in the FastAPI backend.

Outputs (in ./output/deployment/):
  resume_ner_model.pt   — Full PyTorch checkpoint (model + config + labels)
  model_weights.pt      — Weights-only (lighter, for production)
  model_config.json     — Architecture + label map as JSON
  tokenizer/            — BERT tokenizer files
  inference.py          — Drop-in inference helper for the backend
  scorer.py             — CV-vs-JD scoring helper for the backend
  README.md             — Quick-start guide
"""

import os
import json
import torch
from transformers import BertTokenizerFast

from config import ModelConfig, DataConfig, ScoringConfig, get_config, LABEL2ID, ID2LABEL
from model import BertBiLSTMCRF


# =============================================================================
# PyTorch export
# =============================================================================

def export_pytorch(model_path: str, out_dir: str, model_config: ModelConfig) -> str:
    print("Exporting PyTorch model …")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = BertBiLSTMCRF(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Weights-only (smaller, for production)
    w_path = os.path.join(out_dir, "model_weights.pt")
    torch.save(model.state_dict(), w_path)
    print(f"  Weights saved  : {w_path}  ({os.path.getsize(w_path)/1e6:.1f} MB)")

    # Full checkpoint (easier for loading in backend)
    full_path = os.path.join(out_dir, "resume_ner_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
    }, full_path)
    print(f"  Full model     : {full_path}  ({os.path.getsize(full_path)/1e6:.1f} MB)")

    # JSON config
    cfg_path = os.path.join(out_dir, "model_config.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "bert_model_name":    model_config.bert_model_name,
            "bert_hidden_size":   model_config.bert_hidden_size,
            "lstm_hidden_size":   model_config.lstm_hidden_size,
            "lstm_num_layers":    model_config.lstm_num_layers,
            "lstm_dropout":       model_config.lstm_dropout,
            "lstm_bidirectional": model_config.lstm_bidirectional,
            "num_labels":         len(LABEL2ID),
            "label2id":           LABEL2ID,
            "id2label":           {str(k): v for k, v in ID2LABEL.items()},
        }, f, indent=2)
    print(f"  Config saved   : {cfg_path}")

    return full_path


# =============================================================================
# Tokenizer export
# =============================================================================

def export_tokenizer(out_dir: str, model_config: ModelConfig) -> str:
    print("Exporting tokenizer …")
    tok_dir = os.path.join(out_dir, "tokenizer")
    tokenizer = BertTokenizerFast.from_pretrained(model_config.bert_model_name)
    tokenizer.save_pretrained(tok_dir)
    print(f"  Tokenizer saved: {tok_dir}")
    return tok_dir


# =============================================================================
# ONNX export (optional, best-effort)
# =============================================================================

def export_onnx(model_path: str, out_dir: str, model_config: ModelConfig):
    print("Attempting ONNX export …")
    try:
        import onnx
    except ImportError:
        print("  [skip] onnx not installed — pip install onnx")
        return None

    ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
    model = BertBiLSTMCRF(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ids  = torch.randint(0, 30000, (1, 128))
    mask = torch.ones(1, 128, dtype=torch.long)

    onnx_path = os.path.join(out_dir, "resume_ner_model.onnx")
    try:
        torch.onnx.export(
            model, (ids, mask), onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["emissions"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "emissions":      {0: "batch", 1: "seq"},
            },
            opset_version=14,
        )
        print(f"  ONNX saved     : {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"  [warn] ONNX export failed: {e}")
        print("  CRF layers need special handling for ONNX; use PyTorch for inference.")
        return None


# =============================================================================
# Deployment artefact generation
# =============================================================================

INFERENCE_PY = '''"""
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
        tokens = re.findall(r"\\b\\w[\\w\'.+-]*\\b|[^\\w\\s]", text)
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
'''

SCORER_PY = '''"""
HireSense AI — CV-vs-JD Scoring Helper
Integrates with your FastAPI scoring endpoint.
"""
from model import CVScorer
from config import ScoringConfig

_scorer = CVScorer(ScoringConfig())

def score_cv(cv_entities: list, jd_text: str, weights: dict = None) -> dict:
    """
    Args:
        cv_entities : output of ResumeNERPipeline.extract()
        jd_text     : raw job description text from recruiter
        weights     : e.g. {"skill": 0.40, "experience": 0.30,
                             "education": 0.15, "project": 0.10,
                             "achievement": 0.05}
    Returns:
        {
          "overall": float (0-100),
          "breakdown": {
              "skill":       {"score": float, "weight": float, ...},
              "experience":  {...},
              ...
          }
        }
    """
    return _scorer.score(cv_entities, jd_text, weights=weights)
'''

README_MD = """# HireSense AI — Deployment Package

## Files

| File | Purpose |
|------|---------|
| `resume_ner_model.pt` | Full PyTorch checkpoint (model + config + labels) |
| `model_weights.pt` | Weights-only file (smaller) |
| `model_config.json` | Architecture + label map |
| `tokenizer/` | BERT tokenizer files |
| `inference.py` | Backend inference helper |
| `scorer.py` | CV-vs-JD scoring helper |

## Quick Start

```python
from inference import ResumeNERPipeline
from scorer import score_cv

pipeline = ResumeNERPipeline(model_dir="./models", device="cpu")

# Extract entities from a CV
entities = pipeline.extract(cv_text)

# Score against a JD with recruiter-configured weights
result = score_cv(
    cv_entities=entities,
    jd_text=jd_text,
    weights={
        "skill":       0.35,
        "experience":  0.30,
        "education":   0.15,
        "project":     0.10,
        "achievement": 0.10,
    }
)

print(result["overall"])     # e.g. 78.4  (0–100)
print(result["breakdown"])   # per-category scores
```

## Entity Types

| Label | Description | Examples |
|-------|-------------|---------|
| SKILL | Skills & tools (all sectors) | Python, Tally, AutoCAD, Litigation |
| EXP | Job titles / experience | Senior Engineer, Accountant |
| EDU | Education & qualifications | B.Tech, CA, MBBS, LLB |
| PROJ | Projects | Built recommendation engine |
| ACH | Awards, publications, patents | Best Auditor Award 2022 |
| CERT | Certifications | AWS Certified, PMP |
| ORG | Employers, universities | Google, IIT Bombay |
| LOC | Locations | Bangalore, Mumbai |
| DATE | Dates & durations | 2018–Present |
| NAME | Candidate name | Priya Sharma |
| CONTACT | Email, phone | priya@email.com |
| SECTOR | Industry sector tag | Finance, Healthcare |

## Sectors Supported

Tech · Finance · Accounting · HR · Sales · Marketing · Design ·
Architecture · Civil Engineering · Mechanical Engineering ·
Electrical Engineering · Healthcare · Legal · Agriculture ·
Education · Aviation · BPO · Operations · Blockchain · Data Science
"""


def create_deployment_package(out_dir: str):
    print("Creating deployment package …")

    (open(os.path.join(out_dir, "inference.py"), "w")
     .write(INFERENCE_PY))
    print(f"  inference.py   : {os.path.join(out_dir, 'inference.py')}")

    (open(os.path.join(out_dir, "scorer.py"), "w")
     .write(SCORER_PY))
    print(f"  scorer.py      : {os.path.join(out_dir, 'scorer.py')}")

    (open(os.path.join(out_dir, "README.md"), "w")
     .write(README_MD))
    print(f"  README.md      : {os.path.join(out_dir, 'README.md')}")


# =============================================================================
# Main
# =============================================================================

def main():
    model_config, _, data_config, _ = get_config()

    model_path = os.path.join(data_config.model_save_path, "best_model.pt")
    out_dir    = os.path.join(data_config.output_dir, "deployment")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train first with train.py")
        return

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("HireSense AI — Model Export")
    print("=" * 60)

    export_pytorch(model_path, out_dir, model_config)
    export_tokenizer(out_dir, model_config)
    export_onnx(model_path, out_dir, model_config)
    create_deployment_package(out_dir)

    print("\n" + "=" * 60)
    print(f"Deployment package ready at: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# HireSense AI — Training Guide

Train the **BERT + BiLSTM + CRF** model for multi-sector Resume Named Entity Recognition.
The model powers the CV scoring feature of HireSense: extracting SKILL, EXP, EDU, PROJ,
ACH, CERT and other entities from a candidate's CV, then scoring them against a recruiter's
Job Description.

## Model Architecture

```
Input text → BERT-base-uncased (768-dim) → BiLSTM (2 layers, 256 hidden, bidirectional)
          → Linear → CRF (Viterbi decoding) → BIO entity tags
```

## Supported Sectors

The model handles CVs from **all** job sectors, including:

Tech · Data Science · Finance · Accounting · Banking · HR · Sales · Marketing ·
Design · Architecture · Civil Engineering · Mechanical Engineering ·
Electrical Engineering · Healthcare · Legal · Agriculture · Education ·
Aviation · Automobile · Blockchain · BPO · Operations · DevOps · and more.

## Hardware Requirements

| Hardware | Notes |
|----------|-------|
| GPU (recommended) | RTX 4060 Ti (8 GB VRAM) or equivalent. FP16 enabled automatically. |
| CPU (fallback) | Works, but ~10× slower. Use `--freeze_bert` to save memory. |
| RAM | 16 GB+ recommended |
| Storage | ~5 GB (model weights + dataset) |

---

## Quick Start

### 1. Setup Environment

```bash
cd scripts/training

python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

# Install PyTorch with CUDA (adjust cu118 to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Tesseract (for scanned PDF OCR)
# macOS:  brew install tesseract poppler
# Linux:  sudo apt-get install -y tesseract-ocr poppler-utils
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

pip install -r requirements.txt
```

### 2. Download the Dataset

**Kaggle Resume PDF dataset** (primary training data — covers all sectors):

```bash
pip install kaggle
kaggle datasets download -d hadikp/resume-data-pdf -p ./data/kaggle_resume_pdf --unzip
```

Or manually: https://www.kaggle.com/datasets/hadikp/resume-data-pdf

The dataset folder structure should look like:
```
data/kaggle_resume_pdf/
  ├── Accountant/
  ├── Aviation/
  ├── Banking/
  ├── Civil Engineer/
  ├── Data Science/
  ├── Finance/
  ├── HR/
  ├── Legal/
  ├── Mechanical Engineer/
  ...
```

The sector is inferred from the subfolder name and stored in each training example.

### 3. Train

**Recommended (RTX 4060 Ti, 8 GB VRAM):**
```bash
python train.py \
    --data_path ./data/kaggle_resume_pdf \
    --output_dir ./output \
    --batch_size 8 \
    --accumulation_steps 4 \
    --epochs 15 \
    --fp16
```

**Lower VRAM (4–6 GB):**
```bash
python train.py --batch_size 4 --accumulation_steps 8 --fp16
```

**CPU only:**
```bash
python train.py --batch_size 4 --accumulation_steps 4 --freeze_bert
```

**Single-file alternative** (no imports needed):
```bash
python train_local.py --data_path ./data/kaggle_resume_pdf --fp16
```

### 4. Monitor Training

Training prints per-epoch:
- Train loss, Val loss, Val F1
- GPU memory usage
- Early stopping countdown

Expected time: **~1–2 hours** for 15 epochs on an RTX 4060 Ti.

### 5. Evaluate

```bash
python evaluate.py
```

Outputs:
- Overall F1 / Precision / Recall
- Per-entity-type breakdown (SKILL, EXP, EDU, ...)
- Inference latency (mean, P50, P95, P99)
- Qualitative extraction examples across 5 sectors
- Full HireSense scoring workflow demo

### 6. Export for Deployment

```bash
python export_model.py
```

Generates `./output/deployment/` with all backend-ready files.

---

## Output Files

After training, `./output/model/` contains:

| File | Description |
|------|-------------|
| `best_model.pt` | Best checkpoint (by Val F1) |
| `resume_ner_model.pt` | Final model for deployment |
| `model_config.json` | Architecture + label map |
| `tokenizer/` | BERT tokenizer files |

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data_path` | `./data/kaggle_resume_pdf` | Path to resume data |
| `--output_dir` | `./output` | Output directory |
| `--batch_size` | `8` | Per-GPU batch size |
| `--accumulation_steps` | `4` | Gradient accumulation (effective batch = 32) |
| `--epochs` | `15` | Max training epochs |
| `--max_length` | `512` | Max BERT token length |
| `--lstm_hidden` | `256` | BiLSTM hidden size |
| `--bert_lr` | `2e-5` | BERT learning rate |
| `--lstm_lr` | `1e-3` | LSTM/CRF learning rate |
| `--fp16` | auto | Mixed precision (auto-enabled on GPU) |
| `--freeze_bert` | `False` | Freeze BERT weights (saves VRAM) |
| `--patience` | `5` | Early stopping patience |
| `--seed` | `42` | Random seed |

---

## Entity Types

| Label | Description | Examples |
|-------|-------------|---------|
| `SKILL` | Skills & tools (all sectors) | Python, Tally, AutoCAD, Litigation, Agronomy |
| `EXP` | Job titles / experience | Senior Engineer, Chartered Accountant |
| `EDU` | Education & qualifications | B.Tech, CA, MBBS, LLB, MBA |
| `PROJ` | Projects | Built recommendation engine |
| `ACH` | Awards, publications, patents | Best Auditor Award, NeurIPS paper |
| `CERT` | Certifications | AWS Certified, PMP, Docker Certified |
| `ORG` | Employers, universities | Google, IIT Bombay, L&T |
| `LOC` | Locations | Bangalore, Mumbai |
| `DATE` | Dates & durations | 2018–Present, 2023 |
| `NAME` | Candidate name | Priya Sharma |
| `CONTACT` | Email, phone | priya@email.com, +91-9876543210 |
| `SECTOR` | Industry sector | Finance, Healthcare |

---

## HireSense Scoring Integration

The model output flows into the scoring pipeline:

```
CV PDF → text extraction → NER entities → CVScorer(entities, jd_text, weights)
                                              ↓
                                        per-category scores (0–100)
                                        + weighted overall score (0–100)
```

**Recruiter-configurable weights:**
```python
weights = {
    "skill":       0.35,   # % weight on skill match
    "experience":  0.30,   # % weight on experience match
    "education":   0.15,   # % weight on education match
    "project":     0.10,   # % weight on project relevance
    "achievement": 0.10,   # % weight on achievements/certs
}
```

**Final score formula (example):**
```
final_score = 0.70 × cv_score + 0.30 × test_score
```

Both weights are configurable per recruiter via the HireSense web interface.

---

## Troubleshooting

**CUDA Out of Memory**
- Reduce `--batch_size` to 4
- Increase `--accumulation_steps` to 8
- Add `--freeze_bert`

**Slow training on Windows**
- Normal — Windows uses `num_workers=0`. Train on Linux/WSL2 for 2–4× speed.

**Empty PDF extraction**
- Ensure PyMuPDF is installed: `pip install PyMuPDF`
- For scanned PDFs, install Tesseract (see Setup above)

**Low F1 (<0.60)**
- Add more annotated data (NER-annotated corpora)
- Increase `--epochs` and `--patience`
- Verify dataset folder structure (PDFs in subfolders by sector)

---

## Google Colab

1. Upload `HireSense_Training.ipynb` to Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells

---

## Deploy to Backend

```bash
cp output/model/resume_ner_model.pt  ../backend/models/
cp output/model/model_config.json    ../backend/models/
cp -r output/model/tokenizer         ../backend/models/
cp output/deployment/inference.py    ../backend/
cp output/deployment/scorer.py       ../backend/
```

Update `MODEL_DIR` in your FastAPI app to point to `./models/`.

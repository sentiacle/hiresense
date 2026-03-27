# HireSense AI - Training Guide

Train the BERT + BiLSTM + CRF model for Resume Named Entity Recognition.

## Model Architecture

- **BERT-base-uncased**: 768-dim contextual embeddings
- **BiLSTM**: 2 layers, 256 hidden units, bidirectional (512 output)
- **CRF**: Conditional Random Field for sequence labeling

## Hardware Requirements

- **GPU**: NVIDIA RTX 4060 Ti (8GB VRAM) or better
- **RAM**: 16GB+ recommended
- **Storage**: 5GB for model + data

## Quick Start (RTX 4060 Ti)
### 1. Setup Environment

```bash
# Navigate to the training directory
cd scripts/training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install PyTorch with CUDA (for modern GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Poppler (required for PDF processing)
# macOS: brew install poppler
# Linux: sudo apt-get install -y poppler-utils

# Install other dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the Kaggle Resume PDF dataset:
- Go to: https://www.kaggle.com/datasets/hadikp/resume-data-pdf
- Download and extract to `./data/kaggle_resume_pdf/`

Or use Kaggle CLI:
```bash
pip install kaggle
kaggle datasets download -d hadikp/resume-data-pdf -p ./data/kaggle_resume_pdf --unzip
```

### 3. Train the Model

**Recommended settings for a modern GPU (e.g., RTX 4060 Ti). Run this from the `scripts/training` directory:**

```bash
python train.py \
    --data_path ./data/kaggle_resume_pdf \
    --output_dir ./output \
    --batch_size 8 \
    --accumulation_steps 4 \ 
    --epochs 15 \
    --fp16
```

This will:
- Use batch size 8 with gradient accumulation (effective batch 32)
- Enable mixed precision (FP16) for faster training
- Save the best model to `./output/`

### 4. Monitor Training

The script will print:
- Loss and F1 score per epoch
- GPU memory usage
- Early stopping when validation F1 plateaus

Expected training time: ~1-2 hours for 15 epochs on an RTX 4060 Ti.

## Output Files

After training, you'll find in `./output/`:

- `best_model.pt` - Best checkpoint (by validation F1)
- `resume_ner_model.pt` - Final model for deployment
- `model_config.json` - Model configuration
- `tokenizer/` - BERT tokenizer files

## Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data_path` | `./data/kaggle_resume_pdf` | Path to resume data |
| `--output_dir` | `./output` | Output directory |
| `--batch_size` | `8` | Batch size (8 for 8GB VRAM) |
| `--accumulation_steps` | `4` | Gradient accumulation |
| `--epochs` | `15` | Max training epochs |
| `--max_length` | `512` | Max sequence length |
| `--lstm_hidden` | `256` | LSTM hidden size |
| `--bert_lr` | `2e-5` | BERT learning rate |
| `--lstm_lr` | `1e-3` | LSTM/CRF learning rate |
| `--fp16` | `True` | Mixed precision (FP16) |
| `--freeze_bert` | `False` | Freeze BERT layers |
| `--patience` | `5` | Early stopping patience |

## Google Colab Alternative

If you prefer Colab:

1. Upload `HireSense_Training.ipynb` to Google Colab
2. Enable GPU: Runtime > Change runtime type > T4 GPU
3. Run all cells

## Entity Types

The model extracts these entities:

| Label | Description | Example |
|-------|-------------|---------|
| SKILL | Programming skills | Python, React, TensorFlow |
| EXP | Job experience | Senior Software Engineer |
| EDU | Education | Master of Science |
| PROJ | Projects | Built recommendation system |
| ACH | Achievements | Winner of hackathon |
| CERT | Certifications | AWS Certified |
| ORG | Organizations | Google, MIT |
| LOC | Locations | San Francisco, CA |
| DATE | Dates | 2020-2023 |
| NAME | Person names | John Smith |
| CONTACT | Contact info | email@example.com |

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size` to 4
- Increase `--accumulation_steps` to 8
- Enable `--fp16` if not already

### Slow Training on Windows
- This is normal due to `num_workers=0` (required for Windows)
- Training on Linux/WSL2 will be faster

### Low F1 Score
- Ensure you have enough training data (500+ resumes)
- Try more epochs with higher patience
- Add more synthetic data with varied templates

## Using the Trained Model

After training, copy these files to the backend:

```bash
cp output/resume_ner_model.pt ../backend/models/
cp output/model_config.json ../backend/models/
cp -r output/tokenizer ../backend/models/
```

Then update `MODEL_PATH` in `backend/model_loader.py` to point to your model.

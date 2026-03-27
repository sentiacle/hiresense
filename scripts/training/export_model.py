# This file is deprecated. Model exportation is now fully integrated into train.py.

import os
import json
import torch
from transformers import BertTokenizerFast

from config import ModelConfig, DataConfig, get_config, LABEL2ID, ID2LABEL
from model import BertBiLSTMCRF


def export_for_pytorch(
    model_path: str,
    output_dir: str,
    model_config: ModelConfig
):
    """
    Export model for PyTorch inference (FastAPI backend)
    """
    print("Exporting for PyTorch inference...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Create model and load weights
    model = BertBiLSTMCRF(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Save model weights only (smaller file)
    weights_path = os.path.join(output_dir, "model_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"  Saved weights to: {weights_path}")
    
    # Save full checkpoint with config
    full_path = os.path.join(output_dir, "resume_ner_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL
    }, full_path)
    print(f"  Saved full model to: {full_path}")
    
    # Save config as JSON
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "bert_model_name": model_config.bert_model_name,
            "bert_hidden_size": model_config.bert_hidden_size,
            "lstm_hidden_size": model_config.lstm_hidden_size,
            "lstm_num_layers": model_config.lstm_num_layers,
            "lstm_dropout": model_config.lstm_dropout,
            "lstm_bidirectional": model_config.lstm_bidirectional,
            "num_labels": len(LABEL2ID),
            "label2id": LABEL2ID,
            "id2label": {str(k): v for k, v in ID2LABEL.items()}
        }, f, indent=2)
    print(f"  Saved config to: {config_path}")
    
    # Get model size
    model_size = os.path.getsize(full_path) / (1024 * 1024)
    print(f"  Model size: {model_size:.1f} MB")
    
    return full_path


def export_for_onnx(
    model_path: str,
    output_dir: str,
    model_config: ModelConfig
):
    """
    Export model to ONNX format for optimized inference
    Note: CRF layer may need special handling
    """
    print("Exporting to ONNX...")
    
    try:
        import onnx
        from torch.onnx import export as onnx_export
    except ImportError:
        print("  ONNX not installed. Skipping ONNX export.")
        return None
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    model = BertBiLSTMCRF(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 128
    dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Export (Note: CRF may not export cleanly)
    onnx_path = os.path.join(output_dir, "resume_ner_model.onnx")
    
    try:
        onnx_export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["emissions"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "emissions": {0: "batch", 1: "seq_len"}
            },
            opset_version=14
        )
        print(f"  Saved ONNX model to: {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        print("  CRF layer may require custom handling for ONNX export.")
        return None


def export_tokenizer(output_dir: str, model_config: ModelConfig):
    """
    Save tokenizer for deployment
    """
    print("Exporting tokenizer...")
    
    tokenizer = BertTokenizerFast.from_pretrained(model_config.bert_model_name)
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"  Saved tokenizer to: {tokenizer_dir}")
    
    return tokenizer_dir


def create_deployment_package(output_dir: str):
    """
    Create a complete deployment package
    """
    print("\nCreating deployment package...")
    
    # Create README
    readme_content = """# HireSense AI - Resume NER Model

## Files

- `resume_ner_model.pt`: Full PyTorch model with config
- `model_weights.pt`: Model weights only
- `model_config.json`: Model configuration
- `tokenizer/`: BERT tokenizer files

## Usage

```python
import torch
from transformers import BertTokenizerFast

# Load model
checkpoint = torch.load("resume_ner_model.pt", map_location="cpu")
model = BertBiLSTMCRF(checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("./tokenizer")

# Inference
text = "John Doe is a software engineer with 5 years of Python experience"
tokens = text.split()
encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(encoding["input_ids"], encoding["attention_mask"])
    predictions = outputs["predictions"]

# Get entities
entities = model.get_entities(tokens, predictions[0], encoding.word_ids())
print(entities)
```

## Model Architecture

- BERT-base-uncased (768-dim embeddings)
- BiLSTM (2 layers, 256 hidden units, bidirectional)
- CRF layer for sequence labeling

## Entity Types

- SKILL: Programming languages, frameworks, tools
- EXP: Experience, job titles
- EDU: Education, degrees
- PROJ: Projects
- ACH: Achievements, awards
- CERT: Certifications
- ORG: Organizations, companies
- LOC: Locations
- DATE: Dates
- NAME: Person names
- CONTACT: Contact information
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"  Created README at: {readme_path}")
    
    # Create inference script
    inference_script = '''"""
Inference script for Resume NER model
"""
import torch
from transformers import BertTokenizerFast
import sys
sys.path.append(".")
from model import BertBiLSTMCRF
from config import ModelConfig

def load_model(model_path="resume_ner_model.pt", device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", ModelConfig())
    model = BertBiLSTMCRF(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("id2label", {})

def extract_entities(model, tokenizer, text, device="cpu"):
    tokens = text.split()
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
    
    return model.get_entities(tokens, outputs["predictions"][0], encoding.word_ids())

if __name__ == "__main__":
    model, id2label = load_model()
    tokenizer = BertTokenizerFast.from_pretrained("./tokenizer")
    
    text = """
    John Smith - Senior Software Engineer
    Skills: Python, JavaScript, React, TensorFlow, AWS
    Experience: Google (2020-Present), Microsoft (2017-2020)
    Education: MS Computer Science, Stanford University
    """
    
    entities = extract_entities(model, tokenizer, text)
    for e in entities:
        print(f"{e['label']}: {e['text']}")
'''
    
    inference_path = os.path.join(output_dir, "inference.py")
    with open(inference_path, "w") as f:
        f.write(inference_script)
    print(f"  Created inference script at: {inference_path}")


def main():
    """Main export function"""
    model_config, _, data_config = get_config()
    
    model_path = os.path.join(data_config.model_save_path, "best_model.pt")
    output_dir = os.path.join(data_config.output_dir, "deployment")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("HireSense AI - Model Export")
    print("=" * 60)
    
    # Export formats
    export_for_pytorch(model_path, output_dir, model_config)
    export_for_onnx(model_path, output_dir, model_config)
    export_tokenizer(output_dir, model_config)
    create_deployment_package(output_dir)
    
    print("\n" + "=" * 60)
    print(f"Deployment package created at: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

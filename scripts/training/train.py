"""
HireSense AI - Training Script
BERT + BiLSTM + CRF for Resume NER

Optimized for modern GPUs (e.g., RTX 4060 Ti) with:
- Mixed precision training (FP16)
- Gradient accumulation
- Memory-efficient settings
"""

import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score

from config import (
    ModelConfig, TrainingConfig, DataConfig,
    get_config, ID2LABEL, LABEL2ID
)
from dataset import DataProcessor, create_dataloaders
from model import BertBiLSTMCRF


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable for fixed input sizes (faster)


def get_optimizer_and_scheduler(
    model: BertBiLSTMCRF,
    train_config: TrainingConfig,
    num_training_steps: int
):
    """
    Create optimizer with different learning rates for BERT and BiLSTM/CRF
    """
    # Separate parameters
    bert_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bert' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': train_config.bert_learning_rate},
        {'params': other_params, 'lr': train_config.lstm_crf_learning_rate}
    ], weight_decay=train_config.weight_decay, eps=train_config.adam_epsilon)
    
    # Learning rate scheduler with warmup
    num_warmup_steps = int(num_training_steps * train_config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


def train_epoch(
    model: BertBiLSTMCRF,
    train_loader,
    optimizer,
    scheduler,
    train_config: TrainingConfig,
    epoch: int,
    scaler: GradScaler = None,
    accumulation_steps: int = 1
):
    """Train for one epoch with mixed precision and gradient accumulation"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        leave=True
    )
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch["input_ids"].to(train_config.device)
        attention_mask = batch["attention_mask"].to(train_config.device)
        labels = batch["labels"].to(train_config.device)
        
        # Forward pass with mixed precision (FP16)
        if train_config.fp16 and scaler is not None:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"] / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config.max_grad_norm
                )
                
                # Update with scaled gradients
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            # Standard FP32 training
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"] / accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item() * accumulation_steps:.4f}",
            "avg_loss": f"{total_loss / num_batches:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            "gpu_mem": f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
        })
    
    return total_loss / num_batches


def evaluate(
    model: BertBiLSTMCRF,
    eval_loader,
    train_config: TrainingConfig
):
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(train_config.device)
            attention_mask = batch["attention_mask"].to(train_config.device)
            labels = batch["labels"].to(train_config.device)

            outputs = model(input_ids, attention_mask, labels)

            total_loss += outputs["loss"].item()
            num_batches += 1

            predictions = outputs["predictions"]

            for i in range(len(predictions)):
                pred_seq = predictions[i]
                label_seq = labels[i].cpu().numpy()
                mask_seq = attention_mask[i].cpu().numpy()

                true_labels = []
                pred_labels = []
                pred_idx = 0

                for j in range(len(label_seq)):
                    # Keep the same valid-token policy used in model.forward
                    if mask_seq[j] != 1 or label_seq[j] == -100:
                        continue

                    true_labels.append(ID2LABEL[label_seq[j]])

                    if pred_idx < len(pred_seq):
                        pred_labels.append(ID2LABEL[pred_seq[pred_idx]])
                    else:
                        pred_labels.append("O")
                    pred_idx += 1

                if len(true_labels) > 0:
                    all_labels.append(true_labels)
                    all_predictions.append(pred_labels)

    avg_loss = total_loss / num_batches

    # safety check
    if len(all_labels) == 0:
        return {"loss": avg_loss, "f1": 0.0, "predictions": [], "labels": []}

    f1 = f1_score(all_labels, all_predictions)

    return {
        "loss": avg_loss,
        "f1": f1,
        "predictions": all_predictions,
        "labels": all_labels
    }


def train(
    model_config: ModelConfig = None,
    train_config: TrainingConfig = None,
    data_config: DataConfig = None
):
    """Main training function"""
    # Get configs
    if model_config is None or train_config is None or data_config is None:
        model_config, train_config, data_config = get_config()
    
    # Set seed
    set_seed(train_config.seed)
    
    # Create output directory
    os.makedirs(data_config.output_dir, exist_ok=True)
    os.makedirs(data_config.model_save_path, exist_ok=True)
    
    print("=" * 60)
    print("HireSense AI - Resume NER Training")
    print("=" * 60)
    print(f"Device: {train_config.device}")
    print(f"Model: {model_config.bert_model_name} + BiLSTM + CRF")
    print(f"LSTM hidden: {model_config.lstm_hidden_size}")
    print(f"Epochs: {train_config.num_epochs}")
    print(f"Batch size: {train_config.train_batch_size}")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(model_config.bert_model_name)
    
    # Load data
    print("\nLoading data...")
    processor = DataProcessor(data_config)
    train_examples, val_examples, test_examples = processor.load_all_data()
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_examples, val_examples, test_examples,
        tokenizer, train_config
    )
    
    # Create model
    print("\nInitializing model...")
    model = BertBiLSTMCRF(model_config)
    model.to(train_config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and scheduler
    num_training_steps = len(train_loader) * train_config.num_epochs
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, train_config, num_training_steps
    )
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if train_config.fp16 and torch.cuda.is_available() else None
    
    # Gradient accumulation for larger effective batch size
    # Accumulate gradients to simulate a larger batch size if memory is constrained
    accumulation_steps = train_config.gradient_accumulation_steps
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Mixed Precision (FP16): {train_config.fp16 and torch.cuda.is_available()}")
    print(f"  Gradient Accumulation Steps: {accumulation_steps}")
    print(f"  Effective Batch Size: {train_config.train_batch_size * train_config.gradient_accumulation_steps}")
    print("=" * 60)
    
    best_f1 = 0
    patience_counter = 0
    training_history = []
    
    for epoch in range(1, train_config.num_epochs + 1):
        start_time = time.time()
        
        # Train with mixed precision
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            train_config, epoch, scaler, accumulation_steps
        )
        
        # Evaluate
        eval_results = evaluate(model, val_loader, train_config)
        
        epoch_time = time.time() - start_time
        
        # Log results
        print(f"\nEpoch {epoch}/{train_config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {eval_results['loss']:.4f}")
        print(f"  Val F1: {eval_results['f1']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": eval_results["loss"],
            "val_f1": eval_results["f1"]
        })
        
        # Save best model
        if eval_results["f1"] > best_f1:
            best_f1 = eval_results["f1"]
            patience_counter = 0
            
            save_path = os.path.join(
                data_config.model_save_path,
                "best_model.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "f1": best_f1
            }, save_path)
            print(f"  Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= train_config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(
        os.path.join(data_config.model_save_path, "best_model.pt")
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_results = evaluate(model, test_loader, train_config)
    
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test F1: {test_results['f1']:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(
        test_results["labels"],
        test_results["predictions"]
    ))
    
    # Save final model for deployment
    final_save_path = os.path.join(
        data_config.model_save_path,
        "resume_ner_model.pt"
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config,
        "f1": test_results["f1"],
        "label2id": LABEL2ID,
        "id2label": ID2LABEL
    }, final_save_path)
    
    # Save config and tokenizer for deployment
    tokenizer.save_pretrained(data_config.model_save_path)
    import json
    with open(os.path.join(data_config.model_save_path, "model_config.json"), "w") as f:
        json.dump({"bert_model_name": model_config.bert_model_name, "lstm_hidden_size": model_config.lstm_hidden_size, "lstm_num_layers": model_config.lstm_num_layers, "num_labels": len(LABEL2ID), "label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, f)
        
    print(f"\nFinal model saved to: {final_save_path}")
    
    return model, training_history, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Resume NER model")
    parser.add_argument("--data_path", type=str, default="./data/kaggle_resume_pdf",
                        help="Path to the primary resume PDF data directory")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for model and results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size (e.g., 8 for 8GB VRAM)")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--lstm_hidden", type=int, default=256,
                        help="LSTM hidden size")
    parser.add_argument("--bert_lr", type=float, default=2e-5,
                        help="BERT learning rate")
    parser.add_argument("--lstm_lr", type=float, default=1e-3,
                        help="LSTM/CRF learning rate")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--freeze_bert", action="store_true", default=False,
                        help="Freeze BERT layers")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    
    args = parser.parse_args()

    # Get default configs
    model_config, train_config, data_config = get_config()

    # Override configs with CLI args
    data_config.kaggle_pdf_path = args.data_path
    data_config.output_dir = args.output_dir
    data_config.model_save_path = os.path.join(args.output_dir, "model")
    
    train_config.train_batch_size = args.batch_size
    train_config.gradient_accumulation_steps = args.accumulation_steps
    train_config.num_epochs = args.epochs
    train_config.max_seq_length = args.max_length
    train_config.bert_learning_rate = args.bert_lr
    train_config.lstm_crf_learning_rate = args.lstm_lr
    train_config.fp16 = args.fp16 or (True if torch.cuda.is_available() else False) # Default to true if cuda available
    train_config.early_stopping_patience = args.patience
    
    model_config.lstm_hidden_size = args.lstm_hidden
    model_config.freeze_bert = args.freeze_bert

    train(model_config, train_config, data_config)

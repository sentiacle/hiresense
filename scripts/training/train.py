"""
HireSense AI - Training Script
BERT + BiLSTM + CRF for Resume NER
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
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
    torch.backends.cudnn.benchmark = False


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
    epoch: int
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        leave=True
    )
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch["input_ids"].to(train_config.device)
        attention_mask = batch["attention_mask"].to(train_config.device)
        labels = batch["labels"].to(train_config.device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.max_grad_norm
        )
        
        # Update
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / num_batches:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / num_batches


def evaluate(
    model: BertBiLSTMCRF,
    eval_loader,
    train_config: TrainingConfig
):
    """Evaluate model on validation/test set"""
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
            
            # Get predictions
            predictions = outputs["predictions"]
            
            # Convert to label strings for seqeval
            for pred_seq, label_seq, mask in zip(
                predictions,
                labels.cpu().numpy(),
                attention_mask.cpu().numpy()
            ):
                pred_labels = []
                true_labels = []
                
                for pred, label, m in zip(pred_seq, label_seq, mask):
                    if m == 1 and label != -100:
                        pred_labels.append(ID2LABEL[pred])
                        true_labels.append(ID2LABEL[label])
                
                if pred_labels:
                    all_predictions.append(pred_labels)
                    all_labels.append(true_labels)
    
    avg_loss = total_loss / num_batches
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
    os.makedirs(data_config.logs_dir, exist_ok=True)
    
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
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_f1 = 0
    patience_counter = 0
    training_history = []
    
    for epoch in range(1, train_config.num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            train_config, epoch
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
    print(f"\nFinal model saved to: {final_save_path}")
    
    return model, training_history, test_results


if __name__ == "__main__":
    train()

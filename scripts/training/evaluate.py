# This file is deprecated. Evaluation is now fully integrated into train.py.

import os
import json
import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict
from transformers import BertTokenizerFast
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from tqdm import tqdm

from config import (
    ModelConfig, TrainingConfig, DataConfig,
    get_config, ID2LABEL, LABEL2ID, ENTITY_GROUPS
)
from dataset import DataProcessor, create_dataloaders
from model import BertBiLSTMCRF


def evaluate_model(
    model: BertBiLSTMCRF,
    test_loader,
    device: str = "cpu"
) -> Dict:
    """
    Comprehensive model evaluation
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    entity_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids, attention_mask)
            predictions = outputs["predictions"]
            
            # Convert to label sequences
            for pred_seq, label_seq, mask in zip(
                predictions,
                labels.numpy(),
                attention_mask.cpu().numpy()
            ):
                pred_labels = []
                true_labels = []
                
                for pred, label, m in zip(pred_seq, label_seq, mask):
                    if m == 1 and label != -100:
                        pred_label = ID2LABEL[pred]
                        true_label = ID2LABEL[label]
                        pred_labels.append(pred_label)
                        true_labels.append(true_label)
                        
                        # Update entity stats
                        if true_label != "O" and pred_label == true_label:
                            entity_type = true_label.split("-")[1] if "-" in true_label else true_label
                            entity_stats[entity_type]["tp"] += 1
                        elif true_label != "O" and pred_label != true_label:
                            entity_type = true_label.split("-")[1] if "-" in true_label else true_label
                            entity_stats[entity_type]["fn"] += 1
                        if pred_label != "O" and pred_label != true_label:
                            entity_type = pred_label.split("-")[1] if "-" in pred_label else pred_label
                            entity_stats[entity_type]["fp"] += 1
                
                if pred_labels:
                    all_predictions.append(pred_labels)
                    all_labels.append(true_labels)
    
    # Calculate metrics
    results = {
        "overall": {
            "f1": f1_score(all_labels, all_predictions),
            "precision": precision_score(all_labels, all_predictions),
            "recall": recall_score(all_labels, all_predictions)
        },
        "per_entity": {},
        "classification_report": classification_report(all_labels, all_predictions)
    }
    
    # Per-entity metrics
    for entity_type, stats in entity_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results["per_entity"][entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn
        }
    
    return results


def evaluate_inference_speed(
    model: BertBiLSTMCRF,
    tokenizer: BertTokenizerFast,
    device: str = "cpu",
    num_samples: int = 100
) -> Dict:
    """
    Measure inference speed
    """
    import time
    
    model.eval()
    model.to(device)
    
    # Sample texts of varying lengths
    sample_texts = [
        "John Doe is a software engineer with 5 years of experience in Python and JavaScript.",
        "Experienced Machine Learning Engineer with expertise in TensorFlow, PyTorch, and deep learning. " * 3,
        "Senior Full Stack Developer at Google with experience in React, Node.js, AWS, and Kubernetes. " * 5,
        "PhD in Computer Science from MIT with publications in NLP and computer vision. " * 10
    ]
    
    times = []
    
    for _ in range(num_samples):
        text = np.random.choice(sample_texts)
        
        # Tokenize
        start = time.time()
        
        tokens = text.split()
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99)
    }


def evaluate_entity_extraction(
    model: BertBiLSTMCRF,
    tokenizer: BertTokenizerFast,
    device: str = "cpu"
) -> None:
    """
    Qualitative evaluation with sample texts
    """
    model.eval()
    model.to(device)
    
    sample_resumes = [
        """
        John Smith
        Senior Software Engineer | john.smith@email.com | San Francisco, CA
        
        Experience:
        Senior Software Engineer at Google (2020 - Present)
        - Led development of machine learning pipeline using Python and TensorFlow
        - Improved system performance by 40%
        
        Software Engineer at Microsoft (2017 - 2020)
        - Developed backend services using Java and Kubernetes
        
        Education:
        Master of Science in Computer Science, Stanford University (2017)
        Bachelor of Science in Computer Engineering, UC Berkeley (2015)
        
        Skills: Python, Java, TensorFlow, PyTorch, Kubernetes, Docker, AWS, SQL
        
        Certifications: AWS Certified Solutions Architect, Google Cloud Professional
        """,
        
        """
        Emily Chen | Data Scientist
        
        Skills: Machine Learning, Deep Learning, NLP, Python, R, SQL, Spark
        
        Experience:
        Data Scientist at Meta (2021-Present)
        Built recommendation systems using collaborative filtering and neural networks
        
        Education:
        PhD in Statistics, MIT (2021)
        
        Publications: "Attention Mechanisms in Recommendation Systems" - NeurIPS 2020
        """
    ]
    
    print("\n" + "=" * 60)
    print("Entity Extraction Examples")
    print("=" * 60)
    
    for i, resume in enumerate(sample_resumes):
        print(f"\n--- Sample {i + 1} ---")
        
        # Tokenize
        tokens = resume.split()
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        word_ids = encoding.word_ids()
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            predictions = outputs["predictions"][0]
        
        # Extract entities
        entities = model.get_entities(tokens, predictions, word_ids)
        
        # Group by type
        by_type = defaultdict(list)
        for entity in entities:
            by_type[entity["label"]].append(entity["text"])
        
        for entity_type, values in sorted(by_type.items()):
            print(f"  {entity_type}: {', '.join(values[:5])}")


def main():
    """Run full evaluation"""
    model_config, train_config, data_config = get_config()
    
    # Load model
    model_path = os.path.join(data_config.model_save_path, "best_model.pt")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=train_config.device)
    
    model = BertBiLSTMCRF(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    tokenizer = BertTokenizerFast.from_pretrained(model_config.bert_model_name)
    
    # Load test data
    print("Loading test data...")
    processor = DataProcessor(data_config)
    _, _, test_examples = processor.load_all_data()
    
    _, _, test_loader = create_dataloaders(
        [], [], test_examples,
        tokenizer, train_config
    )
    
    # Evaluation
    print("\n" + "=" * 60)
    print("Model Evaluation Results")
    print("=" * 60)
    
    results = evaluate_model(model, test_loader, train_config.device)
    
    print(f"\nOverall Metrics:")
    print(f"  F1 Score: {results['overall']['f1']:.4f}")
    print(f"  Precision: {results['overall']['precision']:.4f}")
    print(f"  Recall: {results['overall']['recall']:.4f}")
    
    print(f"\nPer-Entity Metrics:")
    for entity, metrics in sorted(results["per_entity"].items()):
        print(f"  {entity}:")
        print(f"    F1: {metrics['f1']:.4f}, P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}")
    
    print(f"\nDetailed Report:\n{results['classification_report']}")
    
    # Inference speed
    print("\n" + "=" * 60)
    print("Inference Speed")
    print("=" * 60)
    
    speed_results = evaluate_inference_speed(
        model, tokenizer, train_config.device
    )
    
    print(f"  Mean: {speed_results['mean_ms']:.2f} ms")
    print(f"  Std: {speed_results['std_ms']:.2f} ms")
    print(f"  P50: {speed_results['p50_ms']:.2f} ms")
    print(f"  P95: {speed_results['p95_ms']:.2f} ms")
    
    # Qualitative examples
    evaluate_entity_extraction(model, tokenizer, train_config.device)
    
    # Save results
    results_path = os.path.join(data_config.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "overall": results["overall"],
            "per_entity": results["per_entity"],
            "inference_speed": speed_results
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

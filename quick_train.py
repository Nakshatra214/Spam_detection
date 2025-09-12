#!/usr/bin/env python3
"""
Quick training script for testing with a smaller dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path

# Import our enhanced preprocessing
from train import normalize_text_advanced

def quick_train():
    """Quick training for testing purposes"""
    print("🚀 Quick Training for Testing")
    print("=" * 50)
    
    # Load small sample
    df = pd.read_csv('training_data/enron_data_fraud_labeled.csv', nrows=10000)
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution: {df['Label'].value_counts().to_dict()}")
    
    # Preprocess
    print("\\nPreprocessing text...")
    df['Body_clean'] = df['Body'].apply(lambda x: normalize_text_advanced(str(x), use_stemming=True))
    
    # Remove empty texts
    df = df[df['Body_clean'].str.len() > 0]
    print(f"After preprocessing: {len(df)} samples")
    
    # Prepare data
    X = df['Body_clean']
    y = df['Label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Train
    print("\\nTraining model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\\nResults:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Find threshold that maximizes F1 score
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred_thresh = (y_proba >= threshold).astype(int)
        f1_thresh = f1_score(y_test, y_pred_thresh, zero_division=0)
        if f1_thresh > best_f1:
            best_f1 = f1_thresh
            best_threshold = threshold
    
    print(f"\\nOptimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Save model and threshold
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    
    joblib.dump(pipeline, artifacts_dir / 'model.joblib')
    
    with open(artifacts_dir / 'threshold.json', 'w') as f:
        json.dump({
            'optimal_threshold': best_threshold,
            'best_f1': best_f1,
            'roc_auc': roc_auc
        }, f, indent=2)
    
    # Save metrics
    with open(artifacts_dir / 'metrics.json', 'w') as f:
        json.dump({
            'test': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'optimal_threshold': best_threshold
            }
        }, f, indent=2)
    
    print(f"\\n✅ Model saved to {artifacts_dir / 'model.joblib'}")
    print(f"✅ Threshold saved to {artifacts_dir / 'threshold.json'}")
    print(f"✅ Metrics saved to {artifacts_dir / 'metrics.json'}")
    
    return pipeline, best_threshold

if __name__ == "__main__":
    quick_train()
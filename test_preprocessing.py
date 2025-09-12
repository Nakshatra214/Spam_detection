#!/usr/bin/env python3
"""
Test script to demonstrate enhanced text preprocessing and model capabilities
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from pathlib import Path

# Import our enhanced preprocessing
from train import normalize_text_advanced, extract_spam_features
from detectors import score_severity

def demonstrate_preprocessing():
    """Demonstrate the enhanced text preprocessing capabilities"""
    print("🔍 Enhanced Text Preprocessing Demonstration")
    print("=" * 60)
    
    # Sample texts for demonstration
    test_texts = [
        "URGENT! You have WON $1,000,000! Click here NOW to claim your prize! Limited time offer!",
        "Dear colleague, please find the quarterly report attached. Best regards, John",
        "FREE MONEY! No risk! 100% guaranteed! Call now: 1-800-SCAM-NOW",
        "Meeting reminder: Team standup at 2 PM today in conference room A",
        "CONGRATULATIONS! You are our WINNER! Wire transfer $50,000 to claim jackpot prize!",
        "Hi mom, just wanted to check in and see how you're doing. Love you!"
    ]
    
    print("\\nOriginal vs Preprocessed Text:")
    print("-" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\\n{i}. Original:")
        print(f"   {text}")
        
        # Simple preprocessing
        simple_processed = text.lower().strip()
        
        # Advanced preprocessing
        advanced_processed = normalize_text_advanced(text, use_stemming=True)
        
        print(f"   Simple: {simple_processed}")
        print(f"   Advanced: {advanced_processed}")
        
        # Extract spam features
        spam_features = extract_spam_features(text)
        severity, detection_results = score_severity(text)
        
        print(f"   Spam Features: {spam_features}")
        print(f"   Severity Score: {severity}/10")
        if detection_results['urgency_hits']:
            print(f"   Urgency Indicators: {detection_results['urgency_hits']}")
        if detection_results['suspicious_hits']:
            print(f"   Suspicious Patterns: {detection_results['suspicious_hits'][:3]}...")  # Show first 3


def quick_model_comparison():
    """Quick model comparison on a sample of data"""
    print("\\n\\n🏆 Quick Model Comparison")
    print("=" * 60)
    
    # Load a small sample of data for quick testing
    print("Loading sample data...")
    df = pd.read_csv('training_data/enron_data_fraud_labeled.csv', nrows=5000)
    
    print(f"Sample size: {len(df)}")
    print(f"Label distribution: {df['Label'].value_counts().to_dict()}")
    
    # Preprocess text
    print("\\nApplying advanced preprocessing...")
    df['Body_clean'] = df['Body'].apply(lambda x: normalize_text_advanced(str(x), use_stemming=True))
    
    # Prepare features
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(df['Body_clean'])
    y = df['Label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': MultinomialNB()
    }
    
    results = {}
    
    print("\\nTraining and evaluating models...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"\\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        # Test threshold adjustment
        if hasattr(model, 'predict_proba'):
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            print(f"   Threshold Analysis:")
            for thresh in thresholds:
                y_pred_thresh = (y_proba >= thresh).astype(int)
                precision_thresh = precision_score(y_test, y_pred_thresh, zero_division=0)
                recall_thresh = recall_score(y_test, y_pred_thresh, zero_division=0)
                f1_thresh = f1_score(y_test, y_pred_thresh, zero_division=0)
                print(f"     @{thresh}: P={precision_thresh:.3f}, R={recall_thresh:.3f}, F1={f1_thresh:.3f}")
    
    # Summary
    print("\\n\\n📊 Summary Results:")
    print("-" * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")


def demonstrate_vectorizers():
    """Compare different vectorization approaches"""
    print("\\n\\n🔢 Vectorization Comparison")
    print("=" * 60)
    
    sample_texts = [
        "Free money! Click now!",
        "Meeting at 3pm today",
        "WIN BIG PRIZES! Call 1-800-SCAM"
    ]
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=10, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(sample_texts)
    
    # Count Vectorizer  
    count = CountVectorizer(max_features=10, ngram_range=(1, 2))
    count_matrix = count.fit_transform(sample_texts)
    
    print("TF-IDF Features:", tfidf.get_feature_names_out()[:10])
    print("Count Features:", count.get_feature_names_out()[:10])
    
    print("\\nTF-IDF Matrix:")
    print(tfidf_matrix.toarray())
    print("\\nCount Matrix:")
    print(count_matrix.toarray())


if __name__ == "__main__":
    try:
        demonstrate_preprocessing()
        demonstrate_vectorizers()
        quick_model_comparison()
        
        print("\\n\\n✅ Enhanced preprocessing demonstration completed!")
        print("\\nKey improvements implemented:")
        print("- Advanced text cleaning (HTML, URLs, phone numbers)")
        print("- NLTK tokenization, stemming, and lemmatization")
        print("- Enhanced stopwords removal")
        print("- Multiple ML models (Logistic Regression, SVM, Random Forest)")
        print("- TF-IDF and Count vectorization")
        print("- Threshold tuning and comprehensive evaluation")
        print("- Keyword/heuristic rules integration")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
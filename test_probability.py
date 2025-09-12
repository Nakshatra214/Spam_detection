#!/usr/bin/env python3
"""
Test script to check fraud probability functionality
"""
import json
import sys
from pathlib import Path
import numpy as np
from joblib import load

# Add current directory to path
sys.path.append('.')

from detectors import analyze_text

def test_probability_calculation():
    """Test the fraud probability calculation functionality"""
    print("🔍 Testing Fraud Probability Calculation")
    print("=" * 60)
    
    # Load model and threshold
    artifacts_dir = Path('artifacts')
    model_path = artifacts_dir / 'model.joblib'
    threshold_path = artifacts_dir / 'threshold.json'
    
    if not model_path.exists():
        print("❌ Model not found! Please train the model first.")
        return
    
    print("📂 Loading model and configuration...")
    model = load(model_path)
    print(f"✅ Model loaded: {type(model)}")
    
    # Load threshold
    optimal_threshold = 0.5  # Default
    if threshold_path.exists():
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
            optimal_threshold = threshold_data.get('optimal_threshold', 0.5)
    
    print(f"📊 Optimal threshold: {optimal_threshold:.3f}")
    
    # Test cases with different fraud probabilities
    test_cases = [
        {
            "text": "Hello, how are you doing today?",
            "expected_risk": "LOW",
            "description": "Normal conversation"
        },
        {
            "text": "URGENT! You've won $1,000,000! Click this link immediately to claim your prize before it expires!",
            "expected_risk": "HIGH", 
            "description": "Classic spam with urgency and money"
        },
        {
            "text": "Please verify your account by providing your OTP and bank details immediately. Final notice!",
            "expected_risk": "HIGH",
            "description": "Phishing attempt"
        },
        {
            "text": "Free gift card worth $500! No catch, guaranteed! Call now!",
            "expected_risk": "MEDIUM",
            "description": "Promotional spam"
        },
        {
            "text": "Meeting scheduled for tomorrow at 2 PM in conference room B.",
            "expected_risk": "LOW",
            "description": "Business communication"
        },
        {
            "text": "Click here to update your payment information for your subscription renewal.",
            "expected_risk": "MEDIUM",
            "description": "Suspicious link request"
        }
    ]
    
    print("\n🧪 Testing Different Text Samples")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        expected_risk = test_case["expected_risk"]
        description = test_case["description"]
        
        print(f"\n📝 Test Case {i}: {description}")
        print(f"Text: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        
        try:
            # Get ML prediction probability
            try:
                proba = float(model.predict_proba([text])[0][1])
                probability_method = "predict_proba"
            except AttributeError:
                # Fallback for models without predict_proba
                decision = model.decision_function([text])[0]
                proba = 1 / (1 + np.exp(-decision))  # Sigmoid function
                proba = float(proba)
                probability_method = "decision_function + sigmoid"
            
            # Apply threshold
            label = int(proba >= optimal_threshold)
            prediction = "FRAUD/SPAM" if label == 1 else "LEGITIMATE"
            
            # Get rule-based analysis
            rules = analyze_text(text)
            severity = rules.get('severity', 0)
            
            # Determine alert status
            alert = bool(severity >= 6 or (label == 1 and proba >= 0.9))
            
            print(f"🎯 ML Prediction: {prediction}")
            print(f"📊 Fraud Probability: {proba:.4f} ({proba*100:.1f}%)")
            print(f"🎚️ Method: {probability_method}")
            print(f"⚖️ Threshold: {optimal_threshold:.3f}")
            print(f"🔥 Severity Score: {severity}/10")
            print(f"🚨 Alert Triggered: {'YES' if alert else 'No'}")
            
            # Show detected patterns
            if rules.get('urgency_hits'):
                print(f"⚡ Urgency patterns: {rules['urgency_hits']}")
            if rules.get('suspicious_hits'):
                print(f"🚩 Suspicious patterns: {rules['suspicious_hits']}")
            if rules.get('amounts'):
                print(f"💰 Financial amounts: {rules['amounts']}")
            
            # Risk assessment
            if proba >= 0.8:
                risk_level = "🔴 VERY HIGH"
            elif proba >= 0.6:
                risk_level = "🟠 HIGH"
            elif proba >= 0.4:
                risk_level = "🟡 MEDIUM"
            elif proba >= 0.2:
                risk_level = "🔵 LOW"
            else:
                risk_level = "🟢 VERY LOW"
            
            print(f"🎨 Risk Level: {risk_level}")
            print(f"✅ Expected: {expected_risk} | {'✅ MATCH' if expected_risk.upper() in risk_level else '❌ DIFFERENT'}")
            
        except Exception as e:
            print(f"❌ Error processing text: {e}")
    
    # Summary statistics
    print(f"\n📈 Model Information Summary")
    print("=" * 60)
    print(f"Model Type: {type(model).__name__}")
    print(f"Has predict_proba: {hasattr(model, 'predict_proba')}")
    print(f"Has decision_function: {hasattr(model, 'decision_function')}")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    
    # Load and display metrics if available
    metrics_path = artifacts_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"\n📊 Model Performance Metrics")
        print("-" * 30)
        test_metrics = metrics.get('test', {})
        for metric, value in test_metrics.items():
            if isinstance(value, float):
                print(f"{metric.title()}: {value:.4f}")
            else:
                print(f"{metric.title()}: {value}")

if __name__ == "__main__":
    test_probability_calculation()
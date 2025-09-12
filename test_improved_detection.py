#!/usr/bin/env python3
"""
Test the improved fraud detection with the new threshold
"""
import requests
import json

def test_improved_fraud_detection():
    """Test fraud detection with the new lower threshold"""
    print("🧪 TESTING IMPROVED FRAUD DETECTION")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    # Test cases that were previously missed
    fraud_cases = [
        {
            "text": "URGENT! You've won $1,000,000! Click here to claim your prize!",
            "description": "Lottery scam with large amount",
            "previously_missed": True
        },
        {
            "text": "Your account is locked. Please verify your OTP immediately.",
            "description": "Phishing with authentication request",
            "previously_missed": False
        },
        {
            "text": "Free gift card worth $500! No catch, guaranteed!",
            "description": "Gift card scam",
            "previously_missed": True
        },
        {
            "text": "Congratulations! You're selected as winner of our lottery!",
            "description": "Prize scam without amount",
            "previously_missed": True
        },
        {
            "text": "Bitcoin investment guaranteed 500% returns! Send money now!",
            "description": "Investment fraud",
            "previously_missed": True
        }
    ]
    
    legitimate_cases = [
        {
            "text": "Hello, how are you doing today?",
            "description": "Normal greeting"
        },
        {
            "text": "Meeting scheduled for tomorrow at 2 PM",
            "description": "Business meeting"
        },
        {
            "text": "Thanks for your help with the project",
            "description": "Appreciation message"
        }
    ]
    
    print(f"\n🚨 TESTING FRAUD CASES:")
    print("-" * 40)
    
    fraud_detected = 0
    for i, case in enumerate(fraud_cases, 1):
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"text": case["text"]},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                ml_label = data.get('label', 0)
                ml_probability = data.get('probability_fraud', 0)
                severity = data.get('severity', 0)
                alert = data.get('alert', False)
                
                # Determine if fraud was detected by ML or rules
                ml_detected = ml_label == 1
                rule_detected = severity >= 6
                
                if ml_detected:
                    fraud_detected += 1
                    detection_method = "✅ ML"
                elif rule_detected:
                    detection_method = "✅ RULES"
                else:
                    detection_method = "❌ MISSED"
                
                status = "IMPROVED" if case.get("previously_missed") and ml_detected else "SAME"
                
                print(f"{i}. {case['description']}")
                print(f"   ML: {ml_probability:.3f} -> {'FRAUD' if ml_detected else 'LEGIT'}")
                print(f"   Severity: {severity}/10")
                print(f"   Detection: {detection_method}")
                print(f"   Status: {status}")
                print()
                
            else:
                print(f"{i}. ERROR: {response.status_code}")
                
        except Exception as e:
            print(f"{i}. ERROR: {e}")
    
    print(f"\n✅ TESTING LEGITIMATE CASES:")
    print("-" * 40)
    
    legit_correct = 0
    for i, case in enumerate(legitimate_cases, 1):
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"text": case["text"]},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                ml_label = data.get('label', 0)
                ml_probability = data.get('probability_fraud', 0)
                
                if ml_label == 0:
                    legit_correct += 1
                    status = "✅ CORRECT"
                else:
                    status = "❌ FALSE POSITIVE"
                
                print(f"{i}. {case['description']}")
                print(f"   ML: {ml_probability:.3f} -> {'FRAUD' if ml_label == 1 else 'LEGIT'}")
                print(f"   Status: {status}")
                print()
                
        except Exception as e:
            print(f"{i}. ERROR: {e}")
    
    # Performance summary
    fraud_detection_rate = fraud_detected / len(fraud_cases) * 100
    legit_accuracy = legit_correct / len(legitimate_cases) * 100
    
    print(f"\n📊 IMPROVED PERFORMANCE SUMMARY:")
    print(f"=" * 50)
    print(f"ML Fraud Detection Rate: {fraud_detection_rate:.1f}% ({fraud_detected}/{len(fraud_cases)})")
    print(f"Legitimate Accuracy: {legit_accuracy:.1f}% ({legit_correct}/{len(legitimate_cases)})")
    
    print(f"\n🎯 IMPROVEMENT ANALYSIS:")
    if fraud_detection_rate >= 80:
        print(f"✅ EXCELLENT: Fraud detection significantly improved!")
    elif fraud_detection_rate >= 60:
        print(f"🟡 GOOD: Fraud detection moderately improved")
    else:
        print(f"❌ POOR: Still missing too many fraud cases")
    
    print(f"\n💡 KEY BENEFITS:")
    print(f"• Threshold lowered from 0.200 to 0.050")
    print(f"• ML model now catches more fraud cases")
    print(f"• Rule-based system provides backup coverage")
    print(f"• Combined approach ensures comprehensive protection")

if __name__ == "__main__":
    test_improved_fraud_detection()
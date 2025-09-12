#!/usr/bin/env python3
"""
Test script to check API probability functionality
"""
import requests
import json

def test_api_probability():
    """Test the fraud probability through API calls"""
    print("🌐 Testing API Fraud Probability Functionality")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8000"
    
    # Test health endpoint first
    try:
        print("🏥 Testing health endpoint...")
        health_response = requests.get(f"{base_url}/health", timeout=5)
        health_data = health_response.json()
        print(f"✅ Health check: {health_data}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test cases for API
    test_cases = [
        {
            "text": "Hello, how are you?",
            "description": "Normal text"
        },
        {
            "text": "URGENT! You won $1000000! Click here now!",
            "description": "High-risk spam"
        },
        {
            "text": "Please verify your OTP and bank account details immediately.",
            "description": "Phishing attempt"
        },
        {
            "text": "Free gift card! No catch guaranteed!",
            "description": "Promotional spam"
        }
    ]
    
    print(f"\n🧪 Testing API Predictions")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        description = test_case["description"]
        
        print(f"\n📝 API Test {i}: {description}")
        print(f"Text: \"{text}\"")
        
        try:
            # Call prediction API
            response = requests.post(
                f"{base_url}/predict",
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ API Response:")
                print(f"   Label: {data.get('label')} ({'FRAUD/SPAM' if data.get('label') == 1 else 'LEGITIMATE'})")
                print(f"   Probability: {data.get('probability_fraud', 0):.4f} ({data.get('probability_fraud', 0)*100:.1f}%)")
                print(f"   Severity: {data.get('severity', 0)}/10")
                print(f"   Alert: {'YES' if data.get('alert') else 'No'}")
                
                # Show detected patterns
                rules = data.get('rules', {})
                if rules.get('urgency_hits'):
                    print(f"   Urgency: {rules['urgency_hits']}")
                if rules.get('suspicious_hits'):
                    print(f"   Suspicious: {rules['suspicious_hits']}")
                if rules.get('amounts'):
                    print(f"   Amounts: {rules['amounts']}")
                
                # Risk assessment
                prob = data.get('probability_fraud', 0)
                if prob >= 0.8:
                    risk_level = "🔴 VERY HIGH"
                elif prob >= 0.6:
                    risk_level = "🟠 HIGH"
                elif prob >= 0.4:
                    risk_level = "🟡 MEDIUM"
                elif prob >= 0.2:
                    risk_level = "🔵 LOW"
                else:
                    risk_level = "🟢 VERY LOW"
                
                print(f"   Risk Level: {risk_level}")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    # Test /analyze endpoint (should be same as /predict)
    print(f"\n🔄 Testing /analyze endpoint")
    print("-" * 40)
    
    try:
        response = requests.post(
            f"{base_url}/analyze",
            json={"text": "URGENT! Free money! Click now!"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ /analyze endpoint working")
            print(f"   Probability: {data.get('probability_fraud', 0):.4f}")
            print(f"   Label: {data.get('label')}")
        else:
            print(f"❌ /analyze failed: {response.status_code}")
    except Exception as e:
        print(f"❌ /analyze request failed: {e}")

if __name__ == "__main__":
    test_api_probability()
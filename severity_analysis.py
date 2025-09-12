#!/usr/bin/env python3
"""
Comprehensive Severity Calculation Analysis and Rule Demonstration
"""
import re
from typing import Dict, List, Tuple
from detectors import score_severity, analyze_text

def demonstrate_severity_rules():
    """Demonstrate how severity scoring rules work with examples"""
    print("🔍 SPAM DETECTION SEVERITY CALCULATION RULES")
    print("=" * 80)
    
    print("\n📋 SEVERITY SCORING BREAKDOWN")
    print("-" * 50)
    
    # Rule categories and their weights
    rules = {
        "🚨 URGENCY PHRASES": {
            "weight": 3,
            "examples": ["urgent", "immediately", "asap", "final notice", "expires today"],
            "description": "Words that create false urgency"
        },
        "🚩 SUSPICIOUS PATTERNS": {
            "weight": 4,
            "examples": ["gift card", "wire transfer", "bitcoin", "congratulations", "winner"],
            "description": "Direct spam/fraud indicators"
        },
        "🏦 FINANCIAL ACCOUNTS": {
            "weight": 2,
            "examples": ["account", "bank details", "iban", "routing", "card number"],
            "description": "References to financial accounts"
        },
        "🔐 AUTHENTICATION REQUESTS": {
            "weight": 3,
            "examples": ["otp", "verification code", "2fa", "password", "pin"],
            "description": "Requests for authentication info"
        },
        "🔗 LINKS/URLS": {
            "weight": 1,
            "examples": ["http://", "www.", "bit.ly", "click here"],
            "description": "URLs and suspicious links"
        },
        "📞 PHONE NUMBERS": {
            "weight": 2,
            "examples": ["+1-800-123-4567", "1234567890"],
            "description": "Phone numbers in messages"
        },
        "📧 EMAIL ADDRESSES": {
            "weight": 1,
            "examples": ["spam@example.com", "user@temp.com"],
            "description": "Email addresses mentioned"
        },
        "🌐 SUSPICIOUS DOMAINS": {
            "weight": 5,
            "examples": ["@temp.com", "@10minutemail.", "@throwaway."],
            "description": "Known spam/temporary email domains"
        }
    }
    
    for category, info in rules.items():
        print(f"\n{category}")
        print(f"   Weight: {info['weight']} points per match")
        print(f"   Rule: {info['description']}")
        print(f"   Examples: {', '.join(info['examples'][:3])}...")
    
    # Amount-based scoring
    print(f"\n💰 FINANCIAL AMOUNTS (Dynamic Scoring)")
    print("   Rule: Higher amounts = higher severity")
    amount_tiers = [
        ("$1,000,000+", 6),
        ("$100,000+", 5),
        ("$10,000+", 4),
        ("$1,000+", 3),
        ("$100+", 2),
        ("Under $100", 1)
    ]
    
    for tier, points in amount_tiers:
        print(f"   {tier:<12} = {points} points")
    
    # Text feature bonuses
    print(f"\n📝 TEXT FEATURE BONUSES")
    feature_bonuses = [
        ("CAPS RATIO > 30%", 2, "Indicates shouting/spam"),
        ("EXCLAMATION > 3", 2, "Excessive excitement marks"),
        ("SHORT + SUSPICIOUS", 2, "Brief text with high base score"),
        ("MULTIPLE CATEGORIES", 3, "3+ different suspicious types")
    ]
    
    for feature, points, desc in feature_bonuses:
        print(f"   {feature:<20} = +{points} points ({desc})")

def test_severity_examples():
    """Test severity calculation with real examples"""
    print(f"\n\n🧪 SEVERITY CALCULATION EXAMPLES")
    print("=" * 80)
    
    test_cases = [
        {
            "text": "Hello, how are you today?",
            "expected_range": "0-2",
            "description": "Normal conversation"
        },
        {
            "text": "URGENT! Your account is locked. Please verify your OTP immediately!",
            "expected_range": "8-12",
            "description": "Urgency + Auth request + Caps"
        },
        {
            "text": "Congratulations! You've won $1,000,000! Click here to claim your prize!",
            "expected_range": "15-20",
            "description": "Prize scam with large amount"
        },
        {
            "text": "Please provide your bank details and routing number for wire transfer",
            "expected_range": "8-12",
            "description": "Financial information request"
        },
        {
            "text": "Free gift card worth $500! No catch, guaranteed! Call +1-800-SCAM now!",
            "expected_range": "10-15",
            "description": "Multiple suspicious patterns"
        },
        {
            "text": "Final notice! Pay $50 fee or account suspended. Reply with OTP.",
            "expected_range": "12-16",
            "description": "Urgency + Payment + Auth + Threat"
        },
        {
            "text": "Meeting scheduled for tomorrow at 2 PM",
            "expected_range": "0-1",
            "description": "Legitimate business message"
        },
        {
            "text": "BITCOIN INVESTMENT!!! MAKE $10000 TODAY!!! GUARANTEED RETURNS!!!",
            "expected_range": "18-25",
            "description": "Crypto scam with excessive caps/exclamation"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        text = case["text"]
        expected = case["expected_range"]
        description = case["description"]
        
        print(f"\n📝 Example {i}: {description}")
        print(f"Text: \"{text}\"")
        print(f"Expected Range: {expected}")
        
        # Calculate actual severity
        score, details = score_severity(text)
        
        print(f"🎯 ACTUAL SEVERITY: {score}")
        
        # Show breakdown
        breakdown = []
        if details['urgency_hits']:
            breakdown.append(f"Urgency: {len(details['urgency_hits'])} × 3 = {len(details['urgency_hits']) * 3}")
        if details['suspicious_hits']:
            breakdown.append(f"Suspicious: {len(details['suspicious_hits'])} × 4 = {len(details['suspicious_hits']) * 4}")
        if details['account_hits']:
            breakdown.append(f"Accounts: {len(details['account_hits'])} × 2 = {len(details['account_hits']) * 2}")
        if details['auth_hits']:
            breakdown.append(f"Auth: {len(details['auth_hits'])} × 3 = {len(details['auth_hits']) * 3}")
        if details['links']:
            breakdown.append(f"Links: {len(details['links'])} × 1 = {len(details['links']) * 1}")
        if details['phones']:
            breakdown.append(f"Phones: {len(details['phones'])} × 2 = {len(details['phones']) * 2}")
        if details['amounts']:
            amount_score = 0
            for amount_str in details['amounts']:
                digits = re.sub(r"[^0-9]", "", amount_str)
                if digits:
                    val = int(digits)
                    if val >= 1000000: amount_score += 6
                    elif val >= 100000: amount_score += 5
                    elif val >= 10000: amount_score += 4
                    elif val >= 1000: amount_score += 3
                    elif val >= 100: amount_score += 2
                    else: amount_score += 1
            breakdown.append(f"Amounts: {details['amounts']} = {amount_score}")
        
        # Text feature bonuses
        text_features = details.get('text_features', {})
        bonuses = []
        if text_features.get('caps_ratio', 0) > 0.3:
            bonuses.append(f"High caps (+2)")
        if text_features.get('exclamation_count', 0) > 3:
            bonuses.append(f"Excessive ! (+2)")
        
        if breakdown:
            print(f"📊 Breakdown: {' + '.join(breakdown)}")
        if bonuses:
            print(f"🎁 Bonuses: {', '.join(bonuses)}")
        
        # Show detected patterns
        if details['urgency_hits']:
            print(f"⚡ Urgency Patterns: {details['urgency_hits']}")
        if details['suspicious_hits']:
            print(f"🚩 Suspicious Patterns: {details['suspicious_hits']}")
        if details['amounts']:
            print(f"💰 Financial Amounts: {details['amounts']}")

def create_severity_scale():
    """Create a visual severity scale"""
    print(f"\n\n📊 SEVERITY SCALE & RISK LEVELS")
    print("=" * 80)
    
    severity_levels = [
        (0, 2, "🟢 VERY LOW", "Legitimate communication"),
        (3, 5, "🔵 LOW", "Possibly suspicious, monitor"),
        (6, 10, "🟡 MEDIUM", "Alert threshold - investigate"),
        (11, 15, "🟠 HIGH", "Likely spam/fraud - block/warn"),
        (16, 20, "🔴 VERY HIGH", "Definite fraud - immediate action"),
        (21, 999, "⚫ EXTREME", "Advanced persistent fraud")
    ]
    
    print("Severity Score | Risk Level    | Action Required")
    print("-" * 50)
    
    for min_score, max_score, level, action in severity_levels:
        if max_score == 999:
            range_str = f"{min_score}+"
        else:
            range_str = f"{min_score}-{max_score}"
        print(f"{range_str:<13} | {level:<13} | {action}")
    
    print(f"\n🚨 ALERT THRESHOLD: Severity ≥ 6")
    print(f"🎯 CURRENT SYSTEM: Also triggers alerts for ML probability ≥ 90%")

def analyze_pattern_effectiveness():
    """Analyze which patterns are most effective"""
    print(f"\n\n📈 PATTERN EFFECTIVENESS ANALYSIS")
    print("=" * 80)
    
    # Sample texts representing different fraud types
    fraud_examples = {
        "Prize/Lottery Scam": "Congratulations! You've won $1,000,000 in our lottery! Claim now!",
        "Phishing": "Your account is suspended. Verify your OTP and bank details immediately.",
        "Gift Card Scam": "Buy $500 iTunes gift cards and send codes to unlock your account.",
        "Investment Fraud": "Guaranteed 500% returns on Bitcoin investment! Send money now!",
        "Tech Support": "Your computer is infected! Call +1-800-SCAM for immediate fix.",
        "Romance Scam": "I need $1000 for urgent medical expenses. Please wire transfer.",
        "Fake Invoice": "Payment overdue $250. Click link to avoid account suspension."
    }
    
    print("Fraud Type              | Severity | Top Triggers")
    print("-" * 60)
    
    for fraud_type, text in fraud_examples.items():
        score, details = score_severity(text)
        
        # Find top contributing factors
        contributors = []
        if details['urgency_hits']:
            contributors.append(f"Urgency({len(details['urgency_hits'])})")
        if details['suspicious_hits']:
            contributors.append(f"Suspicious({len(details['suspicious_hits'])})")
        if details['amounts']:
            contributors.append(f"Amounts({len(details['amounts'])})")
        if details['auth_hits']:
            contributors.append(f"Auth({len(details['auth_hits'])})")
        
        contributors_str = ", ".join(contributors[:3])
        print(f"{fraud_type:<22} | {score:<8} | {contributors_str}")

if __name__ == "__main__":
    demonstrate_severity_rules()
    test_severity_examples()
    create_severity_scale()
    analyze_pattern_effectiveness()
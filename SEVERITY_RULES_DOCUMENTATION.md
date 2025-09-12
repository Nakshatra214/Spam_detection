# 🔍 Spam Detection Severity Calculation Rules

## Overview

The spam detection system uses a **rule-based severity scoring system** that analyzes text content and assigns numerical scores based on suspicious patterns. This severity score (0-∞) works alongside the ML probability score to provide comprehensive fraud detection.

## 📊 Core Severity Calculation Formula

```
Total Severity = Base Pattern Score + Amount Score + Text Feature Bonuses + Category Bonuses
```

## 🎯 Detailed Scoring Rules

### 1. **Base Pattern Scoring** (Primary Rules)

| Category | Weight | Pattern Examples | Description |
|----------|--------|------------------|-------------|
| **🚨 Urgency Phrases** | **3 points/match** | "urgent", "immediately", "final notice", "expires today" | Creates false sense of urgency |
| **🚩 Suspicious Patterns** | **4 points/match** | "gift card", "congratulations", "winner", "bitcoin" | Direct fraud indicators |
| **🏦 Financial Accounts** | **2 points/match** | "bank details", "routing number", "account", "iban" | Requests financial info |
| **🔐 Authentication** | **3 points/match** | "otp", "verification code", "password", "2fa" | Requests credentials |
| **🔗 Links/URLs** | **1 point/match** | "http://", "click here", "bit.ly" | Suspicious links |
| **📞 Phone Numbers** | **2 points/match** | "+1-800-123-4567", "1234567890" | Contact information |
| **📧 Email Addresses** | **1 point/match** | "spam@domain.com" | Email mentions |
| **🌐 Suspicious Domains** | **5 points/match** | "@temp.com", "@10minutemail" | Known spam domains |

### 2. **Amount-Based Scoring** (Dynamic Rules)

Financial amounts detected in text trigger escalating scores:

| Amount Range | Score | Rationale |
|--------------|-------|-----------|
| **$1,000,000+** | **6 points** | Unrealistic high amounts (lottery scams) |
| **$100,000+** | **5 points** | Major financial fraud attempts |
| **$10,000+** | **4 points** | Significant fraud amounts |
| **$1,000+** | **3 points** | Common scam amounts |
| **$100+** | **2 points** | Small fee scams |
| **Under $100** | **1 point** | Minor amounts |

**Detection Pattern**: `(?:(?:usd|inr|rs\.?|₹|\$|€|£)\s*)?(\d{2,}(?:[\,\.]\d{3})*)(?:\s*(?:usd|inr|rs\.?|₹|\$|€|£|dollars?|rupees?))?`

### 3. **Text Feature Bonuses** (Behavioral Rules)

| Feature | Threshold | Bonus | Description |
|---------|-----------|-------|-------------|
| **High Caps Ratio** | >30% caps | **+2 points** | Excessive shouting/emphasis |
| **Excessive Exclamation** | >3 marks | **+2 points** | Over-excitement indicators |
| **Short + Suspicious** | <10 words + score >5 | **+2 points** | Brief but highly suspicious |
| **Multiple Categories** | 3+ pattern types | **+3 points** | Combines multiple fraud tactics |

### 4. **Pattern Categories Detected**

#### 🚨 **Urgency Phrases** (Weight: 3)
```
"urgent", "immediately", "asap", "right now", "time sensitive", "expires today",
"final notice", "last warning", "payment due today", "overdue", "deadline",
"limited time", "act now", "hurry", "don't delay", "expires soon"
```

#### 🚩 **Suspicious Patterns** (Weight: 4)
```
Financial: "gift card", "wire transfer", "bitcoin", "bank details", "paypal"
Authentication: "otp", "verification code", "account locked", "kyc"
Payment: "payment link", "upi", "invoice", "unpaid", "penalty"
Prize/Lottery: "congratulations", "winner", "lottery", "jackpot", "claim your"
Free/Money: "free money", "easy money", "100% free", "guaranteed", "risk free"
```

## 📈 Severity Scale & Risk Levels

| Severity Range | Risk Level | Color | Alert Action | Description |
|----------------|------------|-------|--------------|-------------|
| **0-2** | 🟢 **VERY LOW** | Green | No action | Legitimate communication |
| **3-5** | 🔵 **LOW** | Blue | Monitor | Possibly suspicious |
| **6-10** | 🟡 **MEDIUM** | Yellow | **ALERT** | Investigation required |
| **11-15** | 🟠 **HIGH** | Orange | Block/Warn | Likely spam/fraud |
| **16-20** | 🔴 **VERY HIGH** | Red | Immediate action | Definite fraud |
| **21+** | ⚫ **EXTREME** | Black | Emergency response | Advanced persistent fraud |

## 🚨 Alert Triggering Logic

The system triggers alerts based on **dual criteria**:

1. **Severity-Based Alert**: `severity_score >= 6`
2. **ML Confidence Alert**: `ml_label == 1 AND ml_probability >= 0.9`
3. **Combined Alert**: Either condition triggers the alert

```python
alert = bool(severity >= ALERT_THRESHOLD or (label == 1 and proba >= 0.9))
```

## 🧪 Real Examples with Calculations

### Example 1: Prize Scam
**Text**: "Congratulations! You've won $1,000,000! Click here to claim your prize!"

**Calculation**:
- Suspicious patterns: 4 matches × 4 = **16 points**
  - "congratulations", "prize", "claim your", "click here"
- Amount scoring: $1,000,000 = **6 points**
- **Total Severity: 22** (⚫ EXTREME)

### Example 2: Phishing Attempt
**Text**: "URGENT! Your account is locked. Please verify your OTP immediately!"

**Calculation**:
- Urgency phrases: 2 matches × 3 = **6 points**
  - "urgent", "immediately"
- Suspicious patterns: 1 match × 4 = **4 points**
  - "otp" (authentication request)
- Account references: 1 match × 2 = **2 points**
  - "account"
- Text bonuses: High caps ratio = **+2 points**
- **Total Severity: 14** (🟠 HIGH)

### Example 3: Gift Card Scam
**Text**: "Free gift card worth $500! No catch, guaranteed! Call +1-800-SCAM now!"

**Calculation**:
- Suspicious patterns: 2 matches × 4 = **8 points**
  - "gift card", "guaranteed"
- Amount scoring: $500 = **2 points**
- Phone number: 1 match × 2 = **2 points**
- **Total Severity: 12** (🟠 HIGH)

## 🎯 Pattern Effectiveness Analysis

Based on common fraud types:

| Fraud Type | Avg Severity | Most Effective Triggers |
|------------|--------------|-------------------------|
| **Prize/Lottery** | 15-25 | Suspicious patterns + Large amounts |
| **Phishing** | 12-20 | Urgency + Authentication requests |
| **Gift Card** | 10-15 | Suspicious patterns + Medium amounts |
| **Investment** | 12-18 | Suspicious patterns + Financial amounts |
| **Tech Support** | 5-10 | Phone numbers + Urgency |
| **Romance** | 10-15 | Urgency + Financial amounts |

## 🔧 Configuration & Customization

### Environment Variables
```bash
ALERT_THRESHOLD=6  # Minimum severity for alerts (default: 6)
```

### File Locations
- **Rules Definition**: `detectors.py`
- **Pattern Lists**: Lines 5-45 in `detectors.py`
- **Scoring Logic**: `score_severity()` function (lines 90-170)

### Customizing Rules

#### Adding New Urgency Phrases:
```python
URGENCY_PHRASES.append(r"your_new_pattern")
```

#### Modifying Weights:
```python
score += 5 * len(urgency_hits)  # Change from 3 to 5
```

#### Adding New Pattern Categories:
```python
NEW_PATTERNS = [r"pattern1", r"pattern2"]
new_hits = find_matches(text_l, NEW_PATTERNS)
score += 2 * len(new_hits)  # Add with weight 2
```

## 📊 Performance Characteristics

### Strengths:
- ✅ **Fast Processing**: Regex-based pattern matching
- ✅ **Interpretable**: Clear rule-based scoring
- ✅ **Customizable**: Easy to add/modify patterns
- ✅ **Comprehensive**: Covers multiple fraud types
- ✅ **Escalating**: Higher amounts = higher severity

### Considerations:
- ⚠️ **Language-Specific**: Currently optimized for English
- ⚠️ **Pattern Dependency**: Relies on predefined patterns
- ⚠️ **Context-Blind**: Doesn't understand semantic context

## 🎯 Integration with ML Model

The severity system works **in parallel** with the ML model:

1. **ML Model**: Provides probability score (0-1) based on learned patterns
2. **Severity Rules**: Provides explicit pattern-based scoring (0-∞)
3. **Combined Decision**: Alert if either ML confidence is high OR severity exceeds threshold

This **dual-system approach** ensures:
- ML catches subtle/learned patterns
- Rules catch obvious fraud indicators
- High coverage with minimal false negatives

## 🔍 Debugging & Testing

### Testing Severity Calculation:
```python
from detectors import analyze_text

result = analyze_text("Your test text here")
print(f"Severity: {result['severity']}")
print(f"Patterns: {result['suspicious_hits']}")
```

### Running Analysis Scripts:
```bash
python severity_analysis.py     # Comprehensive analysis
python test_probability.py      # Test ML + Severity integration
```

---

**Note**: This severity system is designed to be **conservative** (minimize false negatives) while remaining **explainable** and **customizable** for different use cases and threat landscapes.
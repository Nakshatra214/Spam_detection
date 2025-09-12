import re
from typing import Dict, List, Tuple

# Enhanced spam detection patterns
URGENCY_PHRASES = [
    r"urgent", r"immediately", r"asap", r"right now", r"time sensitive", r"expires today",
    r"do not tell", r"confidential", r"keep this between us", r"secret", r"private",
    r"final notice", r"last warning", r"payment due today", r"overdue", r"deadline",
    r"limited time", r"act now", r"hurry", r"dont delay", r"what are you waiting",
    r"expires soon", r"offer expires", r"time running out"
]

SUSPICIOUS_PATTERNS = [
    # Financial patterns
    r"gift\s*card", r"wire\s*transfer", r"crypto|bitcoin|usdt|ethereum", r"bank\s*details",
    r"prepaid", r"itunes|google\s*play\s*card", r"amazon\s*gift", r"paypal", r"venmo",
    r"western\s*union|moneygram", r"beneficiary\s*name|beneficiary\s*account", r"escrow",
    r"neft|rtgs|imps", r"swift\s*code", r"routing\s*number", r"iban",
    
    # Authentication patterns
    r"one\-time\s*password|otp", r"verification\s*code", r"account\s*lock(ed)?",
    r"kyc|re\-kyc", r"pan\s*card|aadhaar", r"2fa|two.factor", r"security\s*code",
    
    # Payment patterns
    r"payment\s*link", r"click\s*this\s*link", r"upi|paytm|phonepe|gpay|google\s*pay",
    r"invoice|unpaid|overdue|settlement|chargeback|refund|penalty|fine",
    r"subscription\s*fee", r"renewal\s*fee", r"processing\s*fee",
    
    # Prize/lottery patterns
    r"congratulations", r"winner", r"lottery", r"prize", r"jackpot", r"sweepstakes",
    r"claim\s*your", r"you\s*have\s*won", r"selected\s*winner", r"lucky\s*number",
    
    # Free/money patterns
    r"free\s*money", r"easy\s*money", r"make\s*money", r"earn\s*cash", r"get\s*rich",
    r"100%\s*free", r"no\s*cost", r"risk\s*free", r"guaranteed", r"amazing\s*deal",
    
    # Suspicious domains/contacts
    r"suspended\s*account", r"account\s*verification", r"update\s*information",
    r"click\s*here", r"download\s*attachment", r"open\s*attachment"
]

AMOUNT_PATTERN = r"(?:(?:usd|inr|rs\.?|₹|\$|€|£)\s*)?(\d{2,}(?:[\,\.]\d{3})*)(?:\s*(?:usd|inr|rs\.?|₹|\$|€|£|dollars?|rupees?))?"
ACCOUNT_PATTERN = r"(account|acct|iban|ifsc|routing|swift|sort\s*code|card\s*number|debit|credit)"
AUTH_PATTERN = r"(otp|one\-time\s*password|2fa|verification\s*code|pin|password|passcode)"
LINK_PATTERN = r"(http[s]?://\S+|www\.[^\s]+|bit\.ly|tinyurl|shorturl)"
PHONE_PATTERN = r"(\+?\d{1,3}[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}|\d{10})"
EMAIL_PATTERN = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"

# Suspicious domains (common in spam)
SUSPICIOUS_DOMAINS = [
    r"@temp.*\.com", r"@10minutemail\.", r"@guerrillamail\.", r"@mailinator\.",
    r"@throwaway\.", r"@disposable\.", r"@fake.*mail\.", r"@spam.*\."
]


def find_matches(text: str, patterns: List[str]) -> List[str]:
    text_l = text.lower() if isinstance(text, str) else ""
    hits = []
    for p in patterns:
        if re.search(p, text_l):
            hits.append(p)
    return hits


def extract_amounts(text: str) -> List[str]:
    text_l = text.lower() if isinstance(text, str) else ""
    return [m.group(0) for m in re.finditer(AMOUNT_PATTERN, text_l)]


def score_severity(text: str) -> Tuple[int, Dict[str, List[str]]]:
    text_l = text.lower() if isinstance(text, str) else ""
    
    # Extract various patterns
    urgency_hits = find_matches(text_l, URGENCY_PHRASES)
    suspicious_hits = find_matches(text_l, SUSPICIOUS_PATTERNS)
    amounts = extract_amounts(text_l)
    account_hits = [m.group(0) for m in re.finditer(ACCOUNT_PATTERN, text_l)]
    auth_hits = [m.group(0) for m in re.finditer(AUTH_PATTERN, text_l)]
    links = [m.group(0) for m in re.finditer(LINK_PATTERN, text_l)]
    phones = [m.group(0) for m in re.finditer(PHONE_PATTERN, text_l)]
    emails = [m.group(0) for m in re.finditer(EMAIL_PATTERN, text_l)]
    
    # Check for suspicious domains
    suspicious_domains = []
    for domain_pattern in SUSPICIOUS_DOMAINS:
        if re.search(domain_pattern, text_l):
            suspicious_domains.append(domain_pattern)
    
    # Calculate severity score with enhanced weights
    score = 0
    
    # Base pattern scores
    score += 3 * len(urgency_hits)        # Urgency is highly suspicious
    score += 4 * len(suspicious_hits)     # Direct suspicious patterns
    score += 2 * len(account_hits)        # Financial account references
    score += 3 * len(auth_hits)           # Authentication requests
    score += 1 * len(links)               # URLs (lower weight as many legit emails have URLs)
    score += 2 * len(phones)              # Phone numbers in spam
    score += 1 * len(emails)              # Email addresses
    score += 5 * len(suspicious_domains)  # Known spam domains
    
    # Amount-based scoring with enhanced logic
    for amount_str in amounts:
        try:
            digits = re.sub(r"[^0-9]", "", amount_str)
            if digits:
                val = int(digits)
                if val >= 1000000:     # 1M+
                    score += 6
                elif val >= 100000:    # 100K+
                    score += 5
                elif val >= 10000:     # 10K+
                    score += 4
                elif val >= 1000:      # 1K+
                    score += 3
                elif val >= 100:       # 100+
                    score += 2
                else:
                    score += 1
        except Exception:
            pass
    
    # Additional heuristics
    text_features = analyze_text_features(text)
    
    # High caps ratio indicates shouting/spam
    if text_features.get('caps_ratio', 0) > 0.3:
        score += 2
    
    # Excessive exclamation marks
    if text_features.get('exclamation_count', 0) > 3:
        score += 2
    
    # Very short text with suspicious content
    if text_features.get('word_count', 0) < 10 and score > 5:
        score += 2
    
    # Multiple suspicious categories
    suspicious_categories = sum([
        len(urgency_hits) > 0,
        len(suspicious_hits) > 0,
        len(account_hits) > 0,
        len(auth_hits) > 0,
        len(amounts) > 0
    ])
    
    if suspicious_categories >= 3:
        score += 3  # Bonus for multiple suspicious elements
    
    details = {
        'urgency_hits': urgency_hits,
        'suspicious_hits': suspicious_hits,
        'amounts': amounts,
        'account_hits': account_hits,
        'auth_hits': auth_hits,
        'links': links,
        'phones': phones,
        'emails': emails,
        'suspicious_domains': suspicious_domains,
        'text_features': text_features
    }
    
    return score, details


def analyze_text_features(text: str) -> dict:
    """Analyze text features for spam detection"""
    if not isinstance(text, str):
        return {}
    
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    }
    
    return features


def analyze_text(text: str) -> Dict[str, List[str]]:
    score, details = score_severity(text)
    details['severity'] = score
    return details

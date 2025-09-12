import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML and preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve
from joblib import dump

# Text preprocessing
import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag

# Download required NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk_downloads = ['punkt_tab', 'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for download in nltk_downloads:
    try:
        if download == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab/')
        elif download == 'punkt':
            nltk.data.find('tokenizers/punkt/')
        elif download in ['stopwords', 'wordnet']:
            nltk.data.find(f'corpora/{download}')
        else:
            nltk.data.find(f'taggers/{download}')
    except LookupError:
        print(f"Downloading {download}...")
        try:
            nltk.download(download, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {download}: {e}")

# Initialize text preprocessing tools after downloading
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load stopwords with fallback
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: Could not load NLTK stopwords: {e}")
    print("Using basic stopwords fallback...")
    # Basic English stopwords fallback
    stop_words = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', 'should', 'now'
    ])

# Spam keywords for heuristic rules
SPAM_KEYWORDS = {
    'urgency': ['urgent', 'immediately', 'asap', 'hurry', 'expires', 'deadline', 'limited time'],
    'financial': ['free', 'money', 'cash', 'prize', 'winner', 'lottery', 'jackpot', 'million', 'billion'],
    'action': ['click', 'call', 'buy', 'order', 'subscribe', 'download', 'install', 'register'],
    'suspicious': ['guaranteed', 'risk-free', 'no catch', 'amazing', 'incredible', 'unbelievable'],
    'contact': ['contact us', 'reply now', 'call now', 'act now', 'dont delay', 'what are you waiting'],
    'offers': ['offer', 'deal', 'bargain', 'save', 'discount', 'sale', '50% off', '100% free']
}

DATA_PATH_DEFAULT = str(Path('training_data') / 'enron_data_fraud_labeled.csv')
ARTIFACTS_DIR = Path('artifacts')
ARTIFACTS_DIR.mkdir(exist_ok=True)


def clean_html(text: str) -> str:
    """Remove HTML tags from text"""
    if not isinstance(text, str):
        return ""
    try:
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    except Exception:
        return text


def remove_urls(text: str) -> str:
    """Remove URLs from text"""
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub('', text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text"""
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    return email_pattern.sub('', text)


def remove_phone_numbers(text: str) -> str:
    """Remove phone numbers from text"""
    phone_pattern = re.compile(r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b')
    return phone_pattern.sub('', text)


def normalize_text_advanced(text: str, use_stemming: bool = True, use_lemmatization: bool = False) -> str:
    """Advanced text preprocessing with multiple cleaning steps"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = clean_html(text)
        
        # Remove URLs, emails, and phone numbers
        text = remove_urls(text)
        text = remove_emails(text)
        text = remove_phone_numbers(text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Remove punctuation but keep apostrophes for contractions
        text = re.sub(r'[^\w\s\']', '', text)
        
        # Tokenize with fallback for different NLTK versions
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to simple tokenization if NLTK fails
            tokens = text.split()
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Apply stemming or lemmatization with error handling
        if use_stemming:
            try:
                tokens = [stemmer.stem(token) for token in tokens]
            except Exception:
                # Fallback if stemming fails
                pass
        elif use_lemmatization:
            try:
                # Get POS tags for better lemmatization
                pos_tags = pos_tag(tokens)
                tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]
            except Exception:
                # Fallback if lemmatization fails
                pass
        
        # Join tokens back to string
        text = ' '.join(tokens)
        
        return text.strip()
        
    except Exception as e:
        print(f"Warning: Text preprocessing failed: {e}")
        return ""


def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tags to WordNet POS tags for lemmatization"""
    from nltk.corpus import wordnet
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def extract_spam_features(text: str) -> dict:
    """Extract heuristic features for spam detection"""
    text_lower = text.lower()
    features = {}
    
    # Count spam keywords by category
    for category, keywords in SPAM_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword in text_lower)
        features[f'{category}_keywords'] = count
    
    # Text statistics
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    # Check for suspicious patterns
    features['has_phone'] = 1 if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) else 0
    features['has_email'] = 1 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text) else 0
    features['has_url'] = 1 if re.search(r'http[s]?://|www\.', text) else 0
    features['has_money'] = 1 if re.search(r'\$\d+|\d+\s*dollars?|\d+\s*USD', text, re.IGNORECASE) else 0
    
    return features


def normalize_text(text: str) -> str:
    """Backward compatibility function"""
    return normalize_text_advanced(text, use_stemming=True, use_lemmatization=False)


def load_data(csv_path: str) -> pd.DataFrame:
    try:
        # Read CSV with low_memory=False to handle mixed types
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csv_path}: {e}")
    
    # Expecting columns: 'Body', 'Label' (binary 0/1)
    if 'Body' not in df.columns or 'Label' not in df.columns:
        available_cols = df.columns.tolist()
        raise ValueError(f"CSV must contain 'Body' and 'Label' columns. Available columns: {available_cols}")
    
    # Select only required columns and drop rows with missing values
    df = df[['Body', 'Label']].dropna()
    
    # Ensure Label is numeric and contains only 0/1 values
    try:
        df['Label'] = pd.to_numeric(df['Label'])
    except ValueError as e:
        raise ValueError(f"Label column must contain numeric values: {e}")
    
    # Check if labels are binary (0/1)
    unique_labels = df['Label'].unique()
    if not all(label in [0, 1] for label in unique_labels):
        print(f"Warning: Non-binary labels found: {unique_labels}. Converting to binary.")
        df['Label'] = (df['Label'] > 0).astype(int)
    
    # Apply text normalization
    df['Body'] = df['Body'].apply(normalize_text)
    
    # Remove rows with empty body text after normalization
    df = df[df['Body'].str.len() > 0]
    
    if len(df) == 0:
        raise ValueError("No valid samples found after data cleaning")
    
    print(f"Loaded {len(df)} samples with {df['Label'].sum()} positive labels")
    return df


def build_pipeline(model_type: str = 'logistic', vectorizer_type: str = 'tfidf', 
                  max_features: int = 20000, ngram_max: int = 2) -> Pipeline:
    """Build ML pipeline with different vectorizers and models"""
    
    # Choose vectorizer
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features, 
            ngram_range=(1, ngram_max), 
            min_df=2,
            max_df=0.95,  # Remove words that appear in >95% of documents
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
        )
    elif vectorizer_type == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, ngram_max),
            min_df=2,
            max_df=0.95,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
    
    # Choose classifier
    if model_type == 'logistic':
        classifier = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=42,
            solver='liblinear'
        )
    elif model_type == 'svm':
        classifier = SVC(
            kernel='linear',
            class_weight='balanced',
            random_state=42,
            probability=True  # Enable probability predictions
        )
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
    elif model_type == 'naive_bayes':
        classifier = MultinomialNB(
            alpha=1.0  # Laplace smoothing
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    return pipeline


def evaluate_with_threshold(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Evaluate model with custom threshold and return detailed metrics"""
    # Get prediction probabilities
    try:
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    except AttributeError:
        # Some models might not have predict_proba
        y_proba = model.decision_function(X_test)
        # Normalize to [0, 1] range
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    
    # Apply custom threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'threshold': float(threshold)
    }
    
    # Add AUC if possible
    try:
        metrics['auc_roc'] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        metrics['auc_roc'] = None
    
    return metrics, y_proba, y_pred


def find_optimal_threshold(model, X_val, y_val, metric: str = 'f1') -> float:
    """Find optimal threshold based on validation set"""
    try:
        y_proba = model.predict_proba(X_val)[:, 1]
    except AttributeError:
        y_proba = model.decision_function(X_val)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred, zero_division=0)
        else:
            score = accuracy_score(y_val, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold


def evaluate_and_report(y_true, y_pred, y_proba=None, prefix: str = "") -> dict:
    """Enhanced evaluation with detailed metrics and probability analysis"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    # Add AUC-ROC if probabilities are provided
    if y_proba is not None:
        try:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics['auc_roc'] = None
    
    print(f"{prefix}Evaluation Results:")
    print(f"{prefix}  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}  Precision: {metrics['precision']:.4f}")
    print(f"{prefix}  Recall:    {metrics['recall']:.4f}")
    print(f"{prefix}  F1-Score:  {metrics['f1']:.4f}")
    if metrics.get('auc_roc'):
        print(f"{prefix}  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Print classification report
    print(f"{prefix}Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))
    
    return metrics


def compare_models(X_train, X_val, X_test, y_train, y_val, y_test, 
                  max_features: int = 20000, ngram_max: int = 2):
    """Compare different models and vectorizers"""
    models_to_test = [
        ('Logistic Regression + TF-IDF', 'logistic', 'tfidf'),
        ('Logistic Regression + Count', 'logistic', 'count'),
        ('SVM + TF-IDF', 'svm', 'tfidf'),
        ('Random Forest + TF-IDF', 'random_forest', 'tfidf'),
        ('Naive Bayes + TF-IDF', 'naive_bayes', 'tfidf'),
    ]
    
    results = []
    best_model = None
    best_f1 = 0
    
    for name, model_type, vectorizer_type in models_to_test:
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        try:
            # Build and train model
            pipeline = build_pipeline(model_type, vectorizer_type, max_features, ngram_max)
            pipeline.fit(X_train, y_train)
            
            # Find optimal threshold on validation set
            optimal_threshold = find_optimal_threshold(pipeline, X_val, y_val, metric='f1')
            print(f"Optimal threshold: {optimal_threshold:.3f}")
            
            # Evaluate on test set with optimal threshold
            test_metrics, y_proba, y_pred = evaluate_with_threshold(
                pipeline, X_test, y_test, optimal_threshold
            )
            
            # Detailed evaluation
            test_metrics_detailed = evaluate_and_report(
                y_test, y_pred, y_proba, prefix="  "
            )
            
            # Store results
            result = {
                'name': name,
                'model_type': model_type,
                'vectorizer_type': vectorizer_type,
                'threshold': optimal_threshold,
                'metrics': test_metrics_detailed,
                'pipeline': pipeline
            }
            results.append(result)
            
            # Track best model based on F1 score
            if test_metrics_detailed['f1'] > best_f1:
                best_f1 = test_metrics_detailed['f1']
                best_model = result
                
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    return results, best_model


def main():
    parser = argparse.ArgumentParser(description='Enhanced Spam Detection Training')
    parser.add_argument('--data', type=str, default=DATA_PATH_DEFAULT, help='Path to training data CSV')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test set size')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--ngram_max', type=int, default=2, help='Maximum n-gram size')
    parser.add_argument('--max_features', type=int, default=20000, help='Maximum features for vectorizer')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['logistic', 'svm', 'random_forest', 'naive_bayes', 'all'],
                       help='Model type to train')
    parser.add_argument('--vectorizer', type=str, default='tfidf', 
                       choices=['tfidf', 'count'], help='Vectorizer type')
    parser.add_argument('--preprocessing', type=str, default='advanced',
                       choices=['simple', 'advanced'], help='Preprocessing level')
    parser.add_argument('--tune_threshold', action='store_true', 
                       help='Tune classification threshold on validation set')
    args = parser.parse_args()

    print("🚀 Enhanced Spam Detection Training Pipeline")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Preprocessing: {args.preprocessing}")
    print(f"Model(s): {args.model}")
    print(f"Vectorizer: {args.vectorizer}")
    print(f"Max features: {args.max_features}")
    print(f"N-gram range: (1, {args.ngram_max})")
    print("="*60)
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    df = load_data(args.data)

    # Apply advanced preprocessing if requested
    if args.preprocessing == 'advanced':
        print("Applying advanced text preprocessing...")
        df['Body'] = df['Body'].apply(lambda x: normalize_text_advanced(x, use_stemming=True))
    else:
        print("Applying simple text preprocessing...")
        df['Body'] = df['Body'].apply(normalize_text)
    
    # Remove empty texts after preprocessing
    df = df[df['Body'].str.len() > 0]
    print(f"Final dataset size: {len(df)} samples")

    X = df['Body'].values
    y = df['Label'].values
    
    # Display class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"\n📈 Class distribution: {class_dist}")
    print(f"Spam ratio: {class_dist.get(1, 0) / len(y):.3f}")
    
    # Check if we have enough samples for stratified split
    min_class_count = min(counts)
    use_stratify = min_class_count >= 2 and len(X) >= 10
    
    if not use_stratify:
        print("⚠️ Warning: Insufficient samples for stratified split. Using random split.")
        stratify_param = None
    else:
        stratify_param = y

    # Split data
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=stratify_param
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=args.random_state, 
            stratify=y_temp if use_stratify and len(y_temp) >= 2 * len(unique) else None
        )
    except ValueError as e:
        print(f"⚠️ Stratified split failed: {e}")
        print("Falling back to random split...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=args.random_state
        )

    print(f"\n📊 Data splits: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    
    # Check if we have any samples to train on
    if len(X_train) == 0:
        raise ValueError("No training samples available after split")

    # Train models
    if args.model == 'all':
        print("\n🔄 Comparing multiple models...")
        results, best_model = compare_models(
            X_train, X_val, X_test, y_train, y_val, y_test, 
            args.max_features, args.ngram_max
        )
        
        # Print summary
        print("\n" + "="*80)
        print("📋 MODEL COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<30} {'F1-Score':<10} {'Precision':<12} {'Recall':<10} {'Accuracy':<10} {'Threshold':<10}")
        print("-"*80)
        
        for result in results:
            metrics = result['metrics']
            print(f"{result['name']:<30} {metrics['f1']:<10.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['accuracy']:<10.4f} {result['threshold']:<10.3f}")
        
        print(f"\n🏆 Best model: {best_model['name']} (F1: {best_model['metrics']['f1']:.4f})")
        
        # Save best model
        final_pipeline = best_model['pipeline']
        final_metrics = {'best_model': best_model['metrics'], 'all_results': results}
        optimal_threshold = best_model['threshold']
        
    else:
        print(f"\n🔄 Training {args.model} model...")
        pipeline = build_pipeline(args.model, args.vectorizer, args.max_features, args.ngram_max)
        pipeline.fit(X_train, y_train)
        
        # Find optimal threshold if requested
        if args.tune_threshold:
            optimal_threshold = find_optimal_threshold(pipeline, X_val, y_val, metric='f1')
            print(f"Optimal threshold: {optimal_threshold:.3f}")
        else:
            optimal_threshold = 0.5
        
        # Evaluate
        test_metrics, y_proba, y_pred = evaluate_with_threshold(
            pipeline, X_test, y_test, optimal_threshold
        )
        
        final_metrics = evaluate_and_report(y_test, y_pred, y_proba, prefix="")
        final_pipeline = pipeline
    
    # Save artifacts
    print("\n💾 Saving model and metrics...")
    
    # Save metrics
    metrics_path = ARTIFACTS_DIR / 'metrics.json'
    # Ensure metrics are JSON serializable
    serializable_metrics = json.loads(json.dumps(final_metrics, default=str))
    with open(metrics_path, 'w') as f:
        json.dump({
            'metrics': serializable_metrics,
            'optimal_threshold': optimal_threshold,
            'preprocessing': args.preprocessing,
            'model_type': args.model,
            'vectorizer_type': args.vectorizer
        }, f, indent=2)
    
    # Save model
    model_path = ARTIFACTS_DIR / 'model.joblib'
    dump(final_pipeline, model_path)
    
    # Save threshold
    threshold_path = ARTIFACTS_DIR / 'threshold.json'
    with open(threshold_path, 'w') as f:
        json.dump({'optimal_threshold': optimal_threshold}, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Metrics saved to: {metrics_path}")
    print(f"✅ Threshold saved to: {threshold_path}")
    
    # Display final performance
    print("\n" + "="*60)
    print("🎯 FINAL MODEL PERFORMANCE")
    print("="*60)
    if args.model == 'all':
        best_metrics = best_model['metrics']
        print(f"Model: {best_model['name']}")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall:    {best_metrics['recall']:.4f}")
        print(f"F1-Score:  {best_metrics['f1']:.4f}")
        if best_metrics.get('auc_roc'):
            print(f"AUC-ROC:   {best_metrics['auc_roc']:.4f}")
    else:
        print(f"Model: {args.model.title()} + {args.vectorizer.upper()}")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"Precision: {final_metrics['precision']:.4f}")
        print(f"Recall:    {final_metrics['recall']:.4f}")
        print(f"F1-Score:  {final_metrics['f1']:.4f}")
        if final_metrics.get('auc_roc'):
            print(f"AUC-ROC:   {final_metrics['auc_roc']:.4f}")
    
    print("\n🎉 Training completed successfully!")


if __name__ == '__main__':
    main()

Team Name: Coder Hunter
 Nakshatra Gautam(Leader)
 Pranali Joshi
 Ankit Makwana
 Rudra Mehta
 Fenil Solanki

# Advanced Spam Detection System

A comprehensive spam and fraud detection system powered by machine learning with real-time analysis capabilities, web interface, and speech recognition support.

## Features

### Core Detection Capabilities
- Advanced Text Analysis: Sophisticated pattern recognition for spam, fraud, and phishing detection
- Rule-Based Detection: Heuristic rules for urgency phrases, financial patterns, and suspicious content
- Machine Learning Models: Support for multiple ML algorithms (Logistic Regression, SVM, Random Forest, Naive Bayes)
- Threshold Optimization: Automatic threshold tuning for optimal precision-recall balance
- Real-time Analysis: Instant text analysis through REST API

### Web Interface
- Modern UI: Clean, responsive web interface with gradient designs and animations
- Live Speech Recognitio: Real-time transcription and analysis of spoken content
- Alert System: Visual and audio notifications for high-risk content
- Alert Dashboard":" View and manage recent fraud alerts
- Mobile Responsive: Works seamlessly on desktop and mobile devices

### Advanced Features
- Multi-language Support: English text processing with NLTK
- Feature Engineering: TF-IDF vectorization with n-gram support
- Cross-validation: Robust model evaluation with stratified splits
- Probability Scoring: Confidence levels for predictions
- Alert Logging: Persistent storage of high-risk detections

## Project Structure

```
Spam_detection/
├── artifacts/              # Model artifacts and configuration
│   ├── metrics.json        # Model performance metrics
│   ├── threshold.json      # Optimal classification threshold
│   └── model.joblib        # Trained model (generated after training)
├── static/
│   └── index.html          # Web interface
├── detectors.py            # Rule-based detection algorithms
├── service_flask.py        # Flask web service and API
├── train.py               # Comprehensive training pipeline
├── quick_train.py         # Simplified training script
├── test_preprocessing.py  # Text preprocessing tests
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd Spam_detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (automatic on first run)
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

### Training the Model

#### Option 1: Quick Training
```bash
python quick_train.py --data your_training_data.csv
```

#### Option 2: Advanced Training
```bash
python train.py --data your_training_data.csv --model all --tune_threshold
```

**Training Parameters:**
- `--data`: Path to CSV file with 'Body' and 'Label' columns
- `--model`: Choose from `logistic`, `svm`, `random_forest`, `naive_bayes`, or `all`
- `--vectorizer`: `tfidf` or `count`
- `--max_features`: Maximum number of features (default: 20000)
- `--preprocessing`: `simple` or `advanced`
- `--tune_threshold`: Enable threshold optimization

### Running the Web Service

```bash
python service_flask.py
```

The service will start on `http://127.0.0.1:8000`

## Usage

### Web Interface
1. Open your browser to `http://127.0.0.1:8000`
2. Enter or paste suspicious text in the analysis box
3. Click "Analyze Text" or use Ctrl+Enter
4. View detailed results including:
   - Spam/legitimate classification
   - Confidence probability
   - Severity score (0-10)
   - Detected patterns and keywords
   - Financial amounts mentioned

### API Endpoints

#### Analyze Text
```bash
POST /predict
Content-Type: application/json

{
    "text": "Congratulations! You've won $1,000,000! Click here to claim..."
}
```

**Response:**
```json
{
    "label": 1,
    "probability_fraud": 0.95,
    "severity": 8,
    "alert": true,
    "rules": {
        "urgency_hits": ["congratulations"],
        "suspicious_hits": ["winner", "click"],
        "amounts": ["1,000,000"]
    }
}
```

#### Other Endpoints
- `GET /health` - Service health check
- `GET /alerts` - Retrieve recent alerts
- `DELETE /alerts` - Clear alert database

### Speech Recognition
1. Click " Start Microphone" in the web interface
2. Speak naturally - text will be transcribed in real-time
3. Completed sentences are automatically analyzed
4. Click " Stop Recording" to finish

## Configuration

### Environment Variables
- `ALERT_THRESHOLD`: Minimum severity score for alerts (default: 6)

### Model Configuration
The system automatically saves and loads:
- Trained model (`artifacts/model.joblib`)
- Optimal threshold (`artifacts/threshold.json`)
- Performance metrics (`artifacts/metrics.json`)

## Performance

Current model performance on test data:
- **Accuracy**: 96.6%
- **Precision**: 85.7%
- **Recall**: 27.6%
- **F1-Score**: 41.7%
- **AUC-ROC**: 98.2%


## Detection Rules

### Pattern Categories
1. **Urgency Phrases**: "urgent", "immediately", "expires today", "final notice"
2. **Financial Patterns**: Gift cards, wire transfers, cryptocurrency, bank details
3. **Authentication Requests**: OTP, verification codes, account locks
4. **Prize/Lottery**: Congratulations, winner, jackpot, sweepstakes
5. **Suspicious Domains**: Temporary email services, known spam domains

### Severity Scoring
- **0-2**: Very Low Risk
- **3-4**: Low Risk
- **5-6**: Medium Risk (Alert Threshold)
- **7-8**: High Risk
- **9-10**: Critical Risk

## Development

### Data Format
Training data should be a CSV file with columns:
- `Body`: Text content to analyze
- `Label`: Binary labels (0 = legitimate, 1 = spam/fraud)

### Adding New Detection Rules
Edit `detectors.py` to add new patterns:
```python
URGENCY_PHRASES.append(r"new_urgent_pattern")
SUSPICIOUS_PATTERNS.append(r"new_suspicious_pattern")
```

### Testing
```bash
python test_preprocessing.py
```

## Dependencies

- `scikit-learn==1.5.1` - Machine learning algorithms
- `pandas==2.2.2` - Data manipulation
- `numpy==1.26.4` - Numerical computing
- `Flask==3.0.3` - Web framework
- `Flask-Cors==5.0.0` - Cross-origin support
- `nltk==3.8.1` - Natural language processing
- `beautifulsoup4==4.12.2` - HTML parsing
- `wordcloud==1.9.2` - Text visualization
- `joblib==1.4.2` - Model serialization

## Security Considerations

- Never log sensitive personal information
- All text analysis is performed locally
- No data is sent to external services
- Alert logs are stored locally in `artifacts/alerts.jsonl`

## License

This project is available for educational and research purposes. 

### Performance Optimization

- For large datasets: Use `--preprocessing simple`
- For better accuracy: Use `--preprocessing advanced --model all`
- For faster inference: Use `naive_bayes` model

---

# Fake News Detection

## Overview
This project implements a machine learning-based system for detecting fake news articles. The system uses natural language processing and machine learning techniques to classify news articles as either real or fake, providing both the classification and confidence score. The project includes a web interface built with Flask for easy interaction with the model.

## Features
- Real-time fake news detection
- Confidence score for predictions
- Web interface for easy interaction
- Pre-trained model with high accuracy
- Text preprocessing and cleaning
- TF-IDF based feature extraction

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Fake-news-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter a news article text in the text area and click "Predict" to get the classification result.

## Dependencies
The project requires the following packages (specified in requirements.txt):
- flask==2.0.1
- numpy==1.21.0
- pandas==1.3.0
- scikit-learn==0.24.2
- nltk==3.6.2
- kagglehub==0.1.0
- pickle-mixin==1.0.2

## Model Details
- Uses TF-IDF vectorization for text preprocessing
- Implements a machine learning classifier for fake news detection
- Includes text cleaning and preprocessing:
  - Special character removal
  - Lowercase conversion
  - Stopword removal
  - Lemmatization
- Provides confidence scores for predictions

## Project Structure
```
Fake-news-detection/
├── app.py                 # Flask application
├── Fake-news-detector.py  # Core detection logic
├── requirements.txt       # Project dependencies
├── templates/            # HTML templates
│   └── index.html       # Main web interface
├── model.pkl            # Trained model
└── tfidf.pkl           # TF-IDF vectorizer
```

## Performance
The model achieves high accuracy in distinguishing between real and fake news articles. The confidence score helps users understand the reliability of each prediction.

## Contributing
Feel free to submit issues and enhancement requests! 
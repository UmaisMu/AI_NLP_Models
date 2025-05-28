# Movie Review Sentiment Analysis

## Overview
This project implements a sentiment analysis system for movie reviews using machine learning. The system can classify movie reviews as either positive or negative, providing both the sentiment prediction and confidence score. The project includes a web interface built with Flask for easy interaction with the model.

## Features
- Sentiment classification of movie reviews
- Confidence score for predictions
- Web interface for easy interaction
- Pre-trained model with high accuracy
- Real-time analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/UmaisMu/AI_NLP_Models.git
cd Movie-Review-Sentiment-Analysis
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

3. Enter a movie review in the text area and click "Analyze" to get the sentiment prediction.

## Dependencies
The project requires the following packages (specified in requirements.txt):
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- nltk==3.8.1
- flask==2.3.3

## Model Details
- The model is trained on the IMDB dataset
- Uses TF-IDF vectorization for text preprocessing
- Implements a machine learning classifier for sentiment prediction
- Provides confidence scores for predictions

## Project Structure
```
Movie-Review-Sentiment-Analysis/
├── app.py                 # Flask application
├── sentiment_analysis.py  # Core sentiment analysis logic
├── requirements.txt       # Project dependencies
├── templates/            # HTML templates
│   └── index.html       # Main web interface
└── imdb_dataset.csv     # Training dataset
```

## Performance
The model achieves high accuracy and F1 score on the test set. Performance metrics are displayed during model initialization.

## Contributing
Feel free to submit issues and enhancement requests! 

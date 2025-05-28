import os
from flask import Flask, render_template, request
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

app = Flask(__name__)

# Use absolute paths for model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
    tfidf = pickle.load(open(os.path.join(BASE_DIR, 'tfidf.pkl'), 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\W', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Clean and preprocess the text
        cleaned_text = clean_text(text)
        
        # Transform the text using TF-IDF
        text_tfidf = tfidf.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        prediction_proba = model.predict_proba(text_tfidf)[0]
        
        # Get confidence score
        confidence = round(max(prediction_proba) * 100, 2)
        
        # Convert prediction to label
        result = "Real" if prediction == 1 else "Fake"
        
        return render_template('index.html', 
                             prediction=result,
                             confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)

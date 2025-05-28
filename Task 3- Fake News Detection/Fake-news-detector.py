# fake_news_detection.py

import pandas as pd
import numpy as np
import re
import nltk
import pickle
import os
from pathlib import Path
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    exit(1)

def load_dataset():
    try:
        # Download dataset from Kaggle
        print("Downloading dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Load datasets
        fake = pd.read_csv(os.path.join(dataset_path, 'Fake.csv'))
        real = pd.read_csv(os.path.join(dataset_path, 'True.csv'))
        
        # Validate data structure
        required_columns = ['title', 'text']
        for df in [fake, real]:
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset missing required columns: {required_columns}")
        
        # Add labels
        fake['label'] = 0
        real['label'] = 1
        
        # Combine and shuffle
        df = pd.concat([fake, real], axis=0).sample(frac=1).reset_index(drop=True)
        df = df[['title', 'text', 'label']]
        
        # Handle missing values
        df = df.dropna()
        
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\W', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

def main():
    # Load and preprocess data
    print("Loading dataset...")
    df = load_dataset()
    
    print("Preprocessing text...")
    df['text'] = df['text'].apply(clean_text)
    
    # Split Data
    X = df['text']
    y = df['label']
    
    print("Vectorizing text...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Naive Bayes Model
    print("\nTraining Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    
    print("\nNaive Bayes Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_nb))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_nb))
    
    # Train and evaluate Random Forest Model
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(n_jobs=-1)  # Use all available cores
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    print("\nRandom Forest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))
    
    # Save models and vectorizer
    try:
        print("\nSaving models...")
        pickle.dump(nb_model, open('model.pkl', 'wb'))
        pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
        print("Models saved successfully!")
    except Exception as e:
        print(f"Error saving models: {e}")

if __name__ == "__main__":
    main()

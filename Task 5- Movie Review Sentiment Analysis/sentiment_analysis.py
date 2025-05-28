import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
import re
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class MovieReviewSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        self.label_map = {'positive': 1, 'negative': 0}
        self.reverse_label_map = {1: 'positive', 0: 'negative'}
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
            
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Simple tokenization by splitting on whitespace
        tokens = text.split()
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Join tokens back into string
        processed_text = ' '.join(tokens)
        
        if not processed_text.strip():
            raise ValueError("After preprocessing, the text is empty")
            
        return processed_text
    
    def load_and_preprocess_data(self, filepath):
        # Convert to absolute path if it's not already
        if not os.path.isabs(filepath):
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
        # Load the dataset
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_columns = ['review', 'sentiment']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")
        
        # Convert sentiment labels to numeric values
        df['sentiment_numeric'] = df['sentiment'].map(self.label_map)
        
        # Preprocess the reviews
        df['processed_review'] = df['review'].apply(self.preprocess_text)
        
        return df
    
    def train(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
            
        # Transform text data to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        # Calculate F1 score using 'weighted' average
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return accuracy, f1
    
    def predict(self, text):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
            
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        
        # Transform the text using the fitted vectorizer
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_tfidf)
        probability = self.model.predict_proba(text_tfidf)
        
        # Convert numeric prediction back to string label
        sentiment = self.reverse_label_map[prediction[0]]
        
        return sentiment, probability[0]

def main():
    # Initialize the analyzer
    analyzer = MovieReviewSentimentAnalyzer()
    
    # Load and preprocess the data
    try:
        # Get the absolute path to the CSV file
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imdb_dataset.csv')
        print(f"Looking for dataset at: {csv_path}")
        
        df = analyzer.load_and_preprocess_data(csv_path)
        
        # Train the model
        accuracy, f1 = analyzer.train(df['processed_review'], df['sentiment_numeric'])
        
        print(f"Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Example prediction
        test_review = "This movie was absolutely fantastic! I loved every minute of it."
        sentiment, probability = analyzer.predict(test_review)
        print(f"\nExample Prediction:")
        print(f"Review: {test_review}")
        print(f"Predicted Sentiment: {sentiment.capitalize()}")
        print(f"Confidence: {max(probability):.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("\nPlease make sure the IMDb dataset is in the same directory as this script.")
        print(f"Current directory: {os.path.dirname(os.path.abspath(__file__))}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
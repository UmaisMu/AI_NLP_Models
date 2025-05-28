from flask import Flask, request, jsonify, render_template
from sentiment_analysis import MovieReviewSentimentAnalyzer
import os

app = Flask(__name__)
analyzer = None

def initialize_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = MovieReviewSentimentAnalyzer()
        # Load and train the model
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imdb_dataset.csv')
        df = analyzer.load_and_preprocess_data(csv_path)
        accuracy, f1 = analyzer.train(df['processed_review'], df['sentiment_numeric'])
        print(f"Model initialized with accuracy: {accuracy:.4f} and F1 score: {f1:.4f}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if analyzer is None:
            initialize_analyzer()
            
        data = request.get_json()
        if not data or 'review' not in data:
            return jsonify({'error': 'No review provided'}), 400
            
        review = data['review']
        if not review.strip():
            return jsonify({'error': 'Review cannot be empty'}), 400
            
        sentiment, confidence = analyzer.predict(review)
        return jsonify({
            'sentiment': sentiment.capitalize(),
            'confidence': float(confidence[1] if sentiment == 'positive' else confidence[0])
        })
        
    except Exception as e:
        print(f"Error analyzing review: {str(e)}")
        return jsonify({'error': 'Failed to analyze review. Please try again.'}), 500

if __name__ == '__main__':
    initialize_analyzer()
    app.run(debug=True) 
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    print("Downloading required NLTK data...")
    
    # Download punkt
    try:
        nltk.data.find('tokenizers/punkt')
        print("✓ punkt already downloaded")
    except LookupError:
        print("Downloading punkt...")
        nltk.download('punkt')
        print("✓ punkt downloaded")
    
    # Download stopwords
    try:
        nltk.data.find('corpora/stopwords')
        print("✓ stopwords already downloaded")
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords')
        print("✓ stopwords downloaded")
    
    print("\nAll required NLTK data has been downloaded successfully!")

if __name__ == "__main__":
    download_nltk_data() 
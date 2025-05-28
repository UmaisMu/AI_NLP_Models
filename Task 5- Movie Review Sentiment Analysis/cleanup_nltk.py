import shutil
import os
import nltk
import ssl

def cleanup_nltk():
    print("Cleaning up NLTK data...")
    
    # List of possible NLTK data directories
    nltk_data_dirs = [
        os.path.expanduser('~/nltk_data'),
        os.path.join(os.path.dirname(os.__file__), 'nltk_data'),
        'C:/nltk_data',
        'D:/nltk_data',
        'E:/nltk_data'
    ]
    
    # Remove existing NLTK data
    for directory in nltk_data_dirs:
        if os.path.exists(directory):
            try:
                print(f"Removing {directory}")
                shutil.rmtree(directory)
            except Exception as e:
                print(f"Could not remove {directory}: {e}")
    
    # Create a new NLTK data directory
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download required NLTK data
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    print("\nDownloading NLTK data...")
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    
    print("\nNLTK data has been cleaned up and reinstalled successfully!")

if __name__ == "__main__":
    cleanup_nltk() 
# Model Deployment Guide

## Overview
This guide provides instructions for deploying the machine learning models developed in this project. The project includes multiple models for different tasks:
- Movie Review Sentiment Analysis
- Fake News Detection
- Customer Segmentation

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/UmaisMu/AI_NLP_Models.git
cd AI_NLP_Modelss
```

2. Create and activate a virtual environment:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Model Deployment

### 1. Movie Review Sentiment Analysis
```bash
cd "Task 5- Movie Review Sentiment Analysis"
python app.py
```
The model will be accessible at: http://localhost:5000

### 2. Fake News Detection
```bash
cd "Task 3- Fake News Detection"
python app.py
```
The model will be accessible at: http://localhost:5000

### 3. Customer Segmentation
```bash
cd "Task 4-Customer Segmentation"
python customer_segmentation.py
```

## Production Deployment

### Using Gunicorn (Linux/Mac)
1. Install Gunicorn:
```bash
pip install gunicorn
```

2. Deploy the Flask application:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Waitress (Windows)
1. Install Waitress:
```bash
pip install waitress
```

2. Deploy the Flask application:
```bash
waitress-serve --port=5000 app:app
```

## Environment Variables
Create a `.env` file in the project root with the following variables:
```
FLASK_ENV=production
FLASK_APP=app.py
```

## Security Considerations
1. Always use HTTPS in production
2. Implement rate limiting
3. Set up proper authentication if needed
4. Keep dependencies updated
5. Use environment variables for sensitive data

## Monitoring and Maintenance
1. Set up logging:
```python
import logging
logging.basicConfig(filename='app.log', level=logging.INFO)
```

2. Monitor model performance:
- Track prediction accuracy
- Monitor response times
- Set up error alerts

## Troubleshooting
1. If the model fails to load:
   - Check if all required files are present
   - Verify model file paths
   - Check Python version compatibility

2. If the server fails to start:
   - Check if port 5000 is available
   - Verify all dependencies are installed
   - Check log files for errors

## Support
For issues and support, please:
1. Check the existing documentation
2. Review the error logs
3. Create an issue in the repository

## Contributing
Feel free to submit issues and enhancement requests! 

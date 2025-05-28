# Customer Segmentation Using Clustering

## Overview
This project implements customer segmentation using K-means clustering to analyze and group customers based on their demographic and behavioral characteristics. The analysis helps businesses understand their customer base and develop targeted marketing strategies.

## Features
- Customer segmentation using K-means clustering
- Exploratory Data Analysis (EDA)
- Visualization of customer segments
- 2D and 3D cluster visualizations
- Correlation analysis
- Silhouette analysis for optimal cluster selection
- Detailed cluster analysis and insights

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Customer-Segmentation-Using-Clustering
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

1. Run the main script:
```bash
python customer_segmentation.py
```

This will:
- Load and preprocess the customer data
- Perform exploratory data analysis
- Generate visualizations
- Perform clustering analysis
- Save results and visualizations

## Dependencies
The project requires the following packages (specified in requirements.txt):
- pandas>=1.3.0
- numpy>=1.20.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=0.24.0

## Analysis Details
The project performs the following analyses:

1. Exploratory Data Analysis:
   - Distribution of customer age
   - Distribution of annual income
   - Distribution of spending score
   - Gender distribution
   - Correlation analysis

2. Clustering Analysis:
   - K-means clustering with optimal k selection
   - Silhouette analysis
   - 2D and 3D visualizations of clusters
   - Cluster characteristics analysis

## Project Structure
```
Customer-Segmentation-Using-Clustering/
├── customer_segmentation.py  # Main analysis script
├── requirements.txt         # Project dependencies
├── Mall_Customers.csv      # Dataset
├── eda_visualizations.png  # EDA plots
├── correlation_matrix.png  # Correlation heatmap
├── clustering_analysis.png # Clustering analysis plots
├── 2d_clusters.png        # 2D cluster visualization
└── 3d_clusters.png        # 3D cluster visualization
```

## Dataset
The project uses the Mall Customer Segmentation dataset, which includes:
- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

## Output
The script generates several visualization files:
- `eda_visualizations.png`: Basic data distributions
- `correlation_matrix.png`: Feature correlations
- `clustering_analysis.png`: Elbow method and silhouette analysis
- `2d_clusters.png`: 2D view of customer segments
- `3d_clusters.png`: 3D view of customer segments

## Contributing
Feel free to submit issues and enhancement requests! 
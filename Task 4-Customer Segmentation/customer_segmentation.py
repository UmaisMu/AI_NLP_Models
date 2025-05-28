import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')  # Updated style name
sns.set_theme()  # Use seaborn's default theme

def load_data():
    """Load the Mall Customer Segmentation dataset"""
    try:
        print("Attempting to load dataset...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'Mall_Customers.csv')
        print(f"Looking for file at: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        print("Columns in dataset:", df.columns.tolist())
        
        # Rename Genre to Gender for consistency
        if 'Genre' in df.columns:
            df = df.rename(columns={'Genre': 'Gender'})
            print("Renamed 'Genre' column to 'Gender'")
        else:
            print("Warning: 'Genre' column not found in dataset")
            
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print(f"Python version: {sys.version}")
        print(f"Current working directory: {os.getcwd()}")
        return None

def perform_eda(df):
    """Perform Exploratory Data Analysis"""
    try:
        print("\n=== Exploratory Data Analysis ===")
        
        # Display basic information
        print("\nDataset Info:")
        print(df.info())
        
        # Display summary statistics
        print("\nSummary Statistics:")
        print(df.describe())
        
        # Check for missing values
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # Create visualizations
        print("\nCreating EDA visualizations...")
        plt.figure(figsize=(15, 10))
        
        # Distribution of Age
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x='Age', bins=30)
        plt.title('Distribution of Age')
        
        # Distribution of Annual Income
        plt.subplot(2, 2, 2)
        sns.histplot(data=df, x='Annual Income (k$)', bins=30)
        plt.title('Distribution of Annual Income')
        
        # Distribution of Spending Score
        plt.subplot(2, 2, 3)
        sns.histplot(data=df, x='Spending Score (1-100)', bins=30)
        plt.title('Distribution of Spending Score')
        
        # Gender Distribution
        plt.subplot(2, 2, 4)
        sns.countplot(data=df, x='Gender')
        plt.title('Gender Distribution')
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png')
        plt.close()
        print("Saved EDA visualizations to 'eda_visualizations.png'")
        
        # Correlation analysis
        print("\nCreating correlation matrix...")
        plt.figure(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()
        print("Saved correlation matrix to 'correlation_matrix.png'")
        
    except Exception as e:
        print(f"Error in EDA: {str(e)}")
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)

def perform_clustering(df):
    """Perform K-Means clustering"""
    try:
        print("\n=== Performing Clustering Analysis ===")
        # Select features for clustering
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        print("Using features:", features)
        
        # Verify features exist in dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Error: Missing features in dataset: {missing_features}")
            print("Available columns:", df.columns.tolist())
            return df
            
        X = df[features]
        print("Feature matrix shape:", X.shape)
        
        # Standardize the features
        print("Standardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters using elbow method
        print("Finding optimal number of clusters...")
        inertia = []
        silhouette_scores = []
        K = range(2, 11)
        
        for k in K:
            print(f"Testing k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Plot elbow curve
        print("Creating clustering analysis plots...")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        
        # Plot silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(K, silhouette_scores, 'rx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        
        plt.tight_layout()
        plt.savefig('clustering_analysis.png')
        plt.close()
        print("Saved clustering analysis to 'clustering_analysis.png'")
        
        # Perform final clustering with optimal number of clusters
        optimal_k = 5  # Based on elbow method and silhouette analysis
        print(f"\nPerforming final clustering with k={optimal_k}...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        return df
        
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)
        return df

def visualize_clusters(df):
    """Visualize the clusters"""
    try:
        print("\n=== Creating Cluster Visualizations ===")
        # 3D plot of clusters
        print("Creating 3D visualization...")
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(df['Age'], 
                           df['Annual Income (k$)'],
                           df['Spending Score (1-100)'],
                           c=df['Cluster'],
                           cmap='viridis')
        
        ax.set_xlabel('Age')
        ax.set_ylabel('Annual Income (k$)')
        ax.set_zlabel('Spending Score (1-100)')
        plt.title('Customer Segments (3D View)')
        
        # Add colorbar
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig('3d_clusters.png')
        plt.close()
        print("Saved 3D visualization to '3d_clusters.png'")
        
        # 2D plots for different feature combinations
        print("Creating 2D visualizations...")
        plt.figure(figsize=(15, 10))
        
        # Age vs Annual Income
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=df, x='Age', y='Annual Income (k$)', hue='Cluster', palette='viridis')
        plt.title('Age vs Annual Income')
        
        # Age vs Spending Score
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
        plt.title('Age vs Spending Score')
        
        # Annual Income vs Spending Score
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
        plt.title('Annual Income vs Spending Score')
        
        plt.tight_layout()
        plt.savefig('2d_clusters.png')
        plt.close()
        print("Saved 2D visualizations to '2d_clusters.png'")
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)

def analyze_clusters(df):
    """Analyze and interpret the clusters"""
    try:
        print("\n=== Cluster Analysis ===")
        
        # Calculate mean values for each cluster
        cluster_analysis = df.groupby('Cluster').agg({
            'Age': 'mean',
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean',
            'Gender': lambda x: x.value_counts().index[0]
        }).round(2)
        
        print("\nCluster Characteristics:")
        print(cluster_analysis)
        
        # Save cluster analysis to CSV
        cluster_analysis.to_csv('cluster_analysis.csv')
        print("\nSaved cluster analysis to 'cluster_analysis.csv'")
        
        return cluster_analysis
        
    except Exception as e:
        print(f"Error in cluster analysis: {str(e)}")
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)
        return None

def main():
    print("Starting Customer Segmentation Analysis...")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Load data
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    # Perform EDA
    perform_eda(df)
    
    # Perform clustering
    df = perform_clustering(df)
    
    # Visualize clusters
    visualize_clusters(df)
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(df)
    
    print("\nAnalysis complete! Check the generated visualizations and cluster_analysis.csv for detailed results.")

if __name__ == "__main__":
    main()

import pandas as pd #type:ignore
import matplotlib.pyplot as plt #type:ignore
import seaborn as sns #type:ignore
from sklearn.preprocessing import StandardScaler #type:ignore
from sklearn.cluster import KMeans #type:ignore
from sklearn.cluster import AgglomerativeClustering #type:ignore
import scipy.cluster.hierarchy as sch #type:ignore

# Load and preprocess data
def load_and_preprocess_data(file_path):
    # Load dataset from the CSV file
    data = pd.read_csv(file_path)

    # Drop rows with missing values
    data = data.dropna()

    # Convert categorical variables to numerical values (one-hot encoding)
    data = pd.get_dummies(data, drop_first=True)

    # Scale features for better clustering results
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)

    return data, scaled_features

# Perform K-Means clustering
def perform_kmeans_clustering(scaled_features, data):
    # Find the optimal number of clusters using the Elbow Method
    inertia = []
    max_clusters = min(10, len(scaled_features) - 1)
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    # Plot the Elbow Curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    # Fit K-Means with a chosen number of clusters (e.g., 4 clusters)
    optimal_k = min(4, len(scaled_features))
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    clusters = kmeans.fit_predict(scaled_features)

    # Add cluster labels to the dataset
    data['KMeans_Cluster'] = clusters
    return data

# Perform Hierarchical clustering
def perform_hierarchical_clustering(scaled_features, data):
    # Plot the Dendrogram
    plt.figure(figsize=(10, 7))
    sch.dendrogram(sch.linkage(scaled_features, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

    # Fit Hierarchical Clustering with a chosen number of clusters (e.g., 4 clusters)
    optimal_clusters = min(4, len(scaled_features))
    hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
    clusters = hierarchical.fit_predict(scaled_features)

    # Add cluster labels to the dataset
    data['Hierarchical_Cluster'] = clusters
    return data

# Visualize clusters
def visualize_clusters(data):
    # Check if clustering results exist
    if 'KMeans_Cluster' not in data.columns or 'Hierarchical_Cluster' not in data.columns:
        raise ValueError("Cluster columns 'KMeans_Cluster' and 'Hierarchical_Cluster' are missing in the dataset.")

    # Visualize K-Means clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=data['KMeans_Cluster'], palette='viridis')
    plt.title('K-Means Clustering')
    plt.show()

    # Visualize Hierarchical clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=data['Hierarchical_Cluster'], palette='viridis')
    plt.title('Hierarchical Clustering')
    plt.show()

if __name__ == "__main__":
    # Path to your dataset
    file_path = 'E://download//elevate//customer_segmentation//customer_data.csv'
    
    # Load and preprocess the data
    data, scaled_features = load_and_preprocess_data(file_path)
    
    # Perform K-Means clustering
    data_with_kmeans = perform_kmeans_clustering(scaled_features, data)
    print("Data with K-Means Clusters:")
    print(data_with_kmeans.head())
    
    # Perform Hierarchical clustering
    data_with_hierarchical = perform_hierarchical_clustering(scaled_features, data)
    print("Data with Hierarchical Clusters:")
    print(data_with_hierarchical.head())
    
    # Visualize the clusters
    visualize_clusters(data_with_hierarchical)

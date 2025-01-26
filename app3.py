import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Preparation
# File paths
customers_file = 'Customers.csv'
transactions_file = 'Transactions.csv'

# Load datasets
customers_df = pd.read_csv(customers_file)
transactions_df = pd.read_csv(transactions_file)

# Feature engineering
# Aggregate transaction data to compute customer-level metrics
transaction_metrics = transactions_df.groupby('CustomerID').agg(
    TotalSpend=('TotalValue', 'sum'),
    AverageSpend=('TotalValue', 'mean'),
    TransactionCount=('TransactionID', 'count')
).reset_index()

# Merge customer and transaction data
customer_data = pd.merge(customers_df, transaction_metrics, on='CustomerID', how='inner')

# Encode categorical variables (e.g., Region)
encoder = OneHotEncoder(sparse_output=False)
encoded_regions = encoder.fit_transform(customer_data[['Region']])

# Create the final dataset for clustering
features = np.hstack([
    encoded_regions,
    customer_data[['TotalSpend', 'AverageSpend', 'TransactionCount']].values
])

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 2. Clustering Algorithm
# Apply K-Means clustering
num_clusters = 5  # You can adjust this between 2 and 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_features)

# Assign cluster labels to customers
customer_data['Cluster'] = kmeans.labels_

# 3. Clustering Metrics
# Calculate Davies-Bouldin Index and Silhouette Score
db_index = davies_bouldin_score(scaled_features, kmeans.labels_)
silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)

print(f"Davies-Bouldin Index: {db_index}")
print(f"Silhouette Score: {silhouette_avg}")

# 4. Visualization
# Reduce dimensions using PCA for 2D visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(pca_features, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = customer_data['Cluster']

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='tab10', s=50)
plt.title('Customer Clusters (PCA Visualization)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Save cluster assignments for reporting
customer_data[['CustomerID', 'Cluster']].to_csv('Customer_Clusters.csv', index=False)
print("Customer Clusters saved to 'Customer_Clusters.csv'.")

# 5. Deliverables
# Report the results
print(f"Number of Clusters: {num_clusters}")
print(f"Davies-Bouldin Index: {db_index}")
print(f"Silhouette Score: {silhouette_avg}")


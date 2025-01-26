import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# 1. Introduction
# The goal is to identify the top 3 lookalike customers for the first 20 customers (C0001 - C0020) 
# based on customer profiles and transaction history. This notebook documents the steps.

# 2. Data Loading and Exploration
# File paths (update as needed)
customers_file = 'Customers.csv'
products_file = 'Products.csv'
transactions_file = 'Transactions.csv'

# Load datasets
customers_df = pd.read_csv(customers_file)
products_df = pd.read_csv(products_file)
transactions_df = pd.read_csv(transactions_file)

# Display basic info
print("Customers Dataset:")
print(customers_df.head())

print("Products Dataset:")
print(products_df.head())

print("Transactions Dataset:")
print(transactions_df.head())

# 3. Data Merging
transactions_customers = pd.merge(transactions_df, customers_df, on='CustomerID', how='inner')
full_data = pd.merge(transactions_customers, products_df, on='ProductID', how='inner')

# Check merged data
print("Merged Dataset:")
print(full_data.head())

# 4. Feature Engineering
# Use both categorical and numerical features
numerical_columns = ['Price_x', 'Quantity', 'TotalValue']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(full_data[['Region', 'ProductName', 'Category']])

# Scale numerical features
scaler = StandardScaler()
numerical_features = scaler.fit_transform(full_data[numerical_columns])

# Combine all features for similarity calculation
features = np.hstack([encoded_features, numerical_features])

# 5. Similarity Calculation
# Build similarity matrix
similarity_matrix = cosine_similarity(features)

# Get indices for first 20 customers (C0001 to C0020)
customer_indices = [full_data[full_data['CustomerID'] == f"C{str(i).zfill(4)}"].index[0] for i in range(1, 21)]

# Generate lookalikes
lookalike_results = {}
for idx in customer_indices:
    scores = similarity_matrix[idx]
    top_indices = scores.argsort()[-4:-1][::-1]
    lookalikes = [(full_data.iloc[i]['CustomerID'], scores[i]) for i in top_indices if full_data.iloc[i]['CustomerID'] != full_data.iloc[idx]['CustomerID']]
    lookalike_results[full_data.iloc[idx]['CustomerID']] = lookalikes

# 6. Results
# Create Lookalike.csv
lookalike_df = pd.DataFrame({
    'CustomerID': lookalike_results.keys(),
    'Lookalikes': [str(v) for v in lookalike_results.values()]
})

lookalike_df.to_csv('Lookalike.csv', index=False)
print("Lookalike.csv created successfully.")

# Display the generated lookalikes for verification
print("Lookalike Recommendations:")
print(lookalike_df.head())

# 7. Conclusion
# The top 3 lookalikes with their similarity scores for the first 20 customers are saved in Lookalike.csv.
# This can help businesses understand customer similarities and target similar customers effectively.


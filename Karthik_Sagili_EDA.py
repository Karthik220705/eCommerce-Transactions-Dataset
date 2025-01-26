import pandas as pd

# File paths (replace these paths with your actual file paths)
customers_file = 'Customers.csv'
products_file = 'Products.csv'
transactions_file = 'Transactions.csv'

# Load datasets
customers_df = pd.read_csv(customers_file)
products_df = pd.read_csv(products_file)
transactions_df = pd.read_csv(transactions_file)

# Inspect datasets
print("Customers Dataset:")
print(customers_df.head())
print(customers_df.info(), "\n")

print("Products Dataset:")
print(products_df.head())
print(products_df.info(), "\n")

print("Transactions Dataset:")
print(transactions_df.head())
print(transactions_df.info(), "\n")

# Merge datasets
# Merge Transactions with Customers
transactions_customers = pd.merge(transactions_df, customers_df, on='CustomerID', how='inner')

# Merge the above result with Products
full_data = pd.merge(transactions_customers, products_df, on='ProductID', how='inner')

# Preview merged dataset
print("Merged Dataset:")
print(full_data.head())

# Basic EDA
# 1. Summary statistics
print("\nSummary Statistics:")
print(full_data.describe())

# 2. Null values
print("\nMissing Values:")
print(full_data.isnull().sum())

# 3. Unique customers, products, and transactions
print("\nUnique Counts:")
print("Unique Customers:", full_data['CustomerID'].nunique())
print("Unique Products:", full_data['ProductID'].nunique())
print("Unique Transactions:", full_data['TransactionID'].nunique())

# 4. Top 5 selling products
top_products = (
    full_data.groupby('ProductName')['Quantity'].sum()
    .sort_values(ascending=False)
    .head(5)
)
print("\nTop 5 Selling Products:")
print(top_products)

# 5. Sales trend over time
# Convert TransactionDate to datetime
full_data['TransactionDate'] = pd.to_datetime(full_data['TransactionDate'])

# Group by month and calculate total sales
monthly_sales = full_data.resample('M', on='TransactionDate')['TotalValue'].sum()
print("\nMonthly Sales Trend:")
print(monthly_sales)

# Visualization libraries (optional, for creating charts)
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting top-selling products
top_products.plot(kind='bar', title='Top 5 Selling Products')
plt.xlabel('Product Name')
plt.ylabel('Total Quantity Sold')
plt.show()

# Plotting monthly sales trend
plt.figure(figsize=(10, 6))
monthly_sales.plot()
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid()
plt.show()


import pandas as pd
import numpy as np
from io import StringIO

# The dataset as a string (your provided data)
data_string = """5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
...
5.9,3.0,5.1,1.8,Iris-virginica"""

# Convert the string data to a DataFrame
df = pd.read_csv(StringIO(data_string), header=None, 
                 names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Data cleaning
print("Initial data shape:", df.shape)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Remove duplicate rows if any
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed. New shape:", df.shape)

# Basic statistics of numerical columns
print("\nBasic statistics:")
print(df.describe())

# Calculate correlation between numerical features
correlation_matrix = df.iloc[:, :4].corr()

print("\nCorrelation matrix:")
print(correlation_matrix)

# Optionally, you can visualize the correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Iris Dataset Features')
plt.show()

# Get the top 5 correlations
correlations = correlation_matrix.unstack()
correlations = correlations[correlations != 1.0]  # Remove self-correlations
top_5_correlations = correlations.nlargest(5)

print("\nTop 5 correlations:")
for idx, corr in top_5_correlations.items():
    print(f"{idx[0]} - {idx[1]}: {corr:.4f}")

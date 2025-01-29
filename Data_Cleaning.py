import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('House_Data.csv')


# Using Boxplot to visually identify outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.show()

# Step 1: Delete all duplicate data
df = df.drop_duplicates()

# Step 2: Delete all null data
df.isnull().sum()

# Step 3: Delete all outlier data
# Assuming we're using the Interquartile Range (IQR) method for outlier detection

# a statistical technique that identifies outliers in a dataset by calculating the range 
# of the middle 50% of data (between the first and third quartiles) and considering any data points 
# that fall significantly below the lower boundary or above the upper boundary as potential outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal to numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns
for column in numerical_columns:
    df = remove_outliers(df, column)

# Using Boxplot to visually identify outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.show()

# Save the cleaned data to a new CSV file
df.to_csv('Cleaned_House_Data.csv', index=False)

print("Data cleaning completed. Cleaned data saved to 'Cleaned_House_Data.csv'")

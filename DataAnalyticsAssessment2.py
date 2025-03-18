import pandas as pd #Imports the Pandas library for data manipulation and analysis.
import numpy as np #Imports NumPy for numerical operations.
import matplotlib.pyplot as plt #Imports Matplotlib for creating visualizations.
import seaborn as sns #Imports Seaborn, a higher-level interface for visualizations based on Matplotlib.
from sklearn.model_selection import train_test_split #Imports a function to split the dataset into training and testing subsets.
from sklearn.ensemble import RandomForestRegressor #Imports the Random Forest Regressor for regression modeling.
from sklearn.cluster import KMeans #Imports KMeans for clustering analysis.
from sklearn.metrics import mean_squared_error, silhouette_score, r2_score #Imports metrics for evaluating regression and clustering models.

# Load the dataset from a file path
dataset = pd.read_csv('dataset for assignment 2.csv') #Loads the dataset from a CSV file into a Pandas DataFrame.

# Display basic information about the dataset
print("Dataset Overview:") #Prints a header for dataset information.
print(dataset.info()) #Displays basic dataset information, including column names, data types, and non-null counts.
print("\nFirst 5 rows of the dataset:") #Prints a header for the first five rows of the dataset.
print(dataset.head()) #Displays the first five rows of the dataset.

# Handle missing values (if any)
print("\nHandling Missing Values...") #Indicates the start of missing value handling.
dataset.fillna(method='ffill', inplace=True)  #Fills missing values using forward fill (propagating the last valid value)
print("Missing values handled.") #Confirms that missing values are handled.

# Exploratory Data Analysis (EDA)
print("\nPerforming Exploratory Data Analysis...") #Indicates the start of EDA.
print("Summary Statistics:") #Prints a header for summary statistics.
print(dataset.describe()) #Displays summary statistics (mean, standard deviation) for numeric columns.

# Visualize correlations between numeric features only
plt.figure(figsize=(10, 8)) #Sets the figure size for the heatmap.

# Select only numeric columns for correlation
numeric_data = dataset.select_dtypes(include=[np.number])  #Selects only numeric columns from the dataset.
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f") #Creates a heatmap to visualize correlations between numeric features.
plt.title("Correlation Heatmap for numeric values") #Sets the title of the heatmap.
plt.show() #Displays the heatmap.

# Distribution of App Sessions
sns.histplot(dataset['App Sessions'], kde=True) #Creates a histogram with a KDE (Kernel Density Estimate) for the 'App Sessions' column.
plt.title("Distribution of App Sessions") #Sets the title of the plot.
plt.xlabel("App Sessions") #Labels the x-axis.
plt.ylabel("Frequency") #Labels the y-axis.
plt.show() #Displays the plot.

# Scatter plot for Distance Travelled vs Calories Burned
plt.figure(figsize=(8, 6)) #Sets the figure size for the scatter plot.
sns.scatterplot(data=dataset, x='Distance Travelled (km)', y='Calories Burned', hue='Activity Level') #Creates a scatter plot with points colored by 'Activity Level'.
plt.title("Distance Travelled vs Calories Burned") #Sets the title of the plot.
plt.xlabel("Distance Travelled (km)") #Labels the x-axis.
plt.ylabel("Calories Burned") #Labels the y-axis.
plt.show() #Displays the plot.

# Feature Engineering
print("\nCreating new features...") #Indicates the start of feature engineering.
dataset['Calories_per_Session'] = dataset['Calories Burned'] / dataset['App Sessions'] #Creates a new feature for calories burned per session.
dataset['Distance_per_Session'] = dataset['Distance Travelled (km)'] / dataset['App Sessions'] #Creates a new feature for distance traveled per session.
print("New features created.")  # Confirms that new features are created.

# Select features and target for regression
features = ['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned', 'Calories_per_Session', 'Distance_per_Session'] #Defines the list of features to use for regression.
target = 'Calories Burned' #Defines the target variable.

X = dataset[features] #Selects the feature columns for the regression model.
y = dataset[target] #Selects the target variable.

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Splits the dataset into training and testing subsets (80%-20% split).

# Regression Model: Random Forest Regressor
print("\nBuilding Regression Model...") #Indicates the start of regression modeling.
regressor = RandomForestRegressor(random_state=42) #Initializes the Random Forest Regressor.
regressor.fit(X_train, y_train) #Trains the regressor on the training data.
y_pred = regressor.predict(X_test) #Predicts the target variable for the test set.

# Evaluate Regression Model
mse = mean_squared_error(y_test, y_pred) #Calculates the Mean Squared Error (MSE).
r2 = r2_score(y_test, y_pred) #Calculates the R-squared (R2) score.
print(f"Mean Squared Error (MSE): {mse}") #Prints the MSE.
print(f"R-squared (R2): {r2}") #Prints the R-squared score.

# Clustering: KMeans to identify user groups
print("\nPerforming Clustering...") #Indicates the start of clustering analysis.
clustering_features = ['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned'] #Defines the features to use for clustering.
kmeans = KMeans(n_clusters=3, random_state=42) #Initializes KMeans with 3 clusters.
clusters = kmeans.fit_predict(dataset[clustering_features]) #Fits KMeans and assigns cluster labels to each row.

# Add cluster labels to the dataset
dataset['Cluster'] = clusters #Adds the cluster labels to the dataset.

# Evaluate Clustering with Silhouette Score
silhouette_avg = silhouette_score(dataset[clustering_features], clusters) #Calculates the Silhouette Score for clustering quality.
print(f"Silhouette Score: {silhouette_avg}") #Prints the Silhouette Score.

# Visualize Clusters
plt.figure(figsize=(8, 6)) #Sets the figure size for the cluster visualization.
sns.scatterplot(data=dataset, x='Distance Travelled (km)', y='Calories Burned', hue='Cluster', palette='viridis') #Creates a scatter plot with points colored by cluster labels.
plt.title("Clustering of Users") #Sets the title of the plot.
plt.xlabel("Distance Travelled (km)") #Labels the axes.
plt.ylabel("Calories Burned") #Labels the axes.
plt.legend(title="Cluster") #Adds a legend for clusters.
plt.show() #Displays the plot.

# Save the modified dataset with clusters
dataset.to_csv('clustered_dataset.csv', index=False) #Saves the dataset with cluster labels to a CSV file.
print("Clustered dataset saved as 'clustered_dataset.csv'.") #Confirms that the file is saved.

# Box Plot: Distribution of Calories Burned by Activity Level
plt.figure(figsize=(8, 6)) #Sets the figure size for the box plot.
sns.boxplot(x='Activity Level', y='Calories Burned', data=dataset, palette="Set3") #Creates a box plot for calories burned grouped by activity level.
plt.title("Calories Burned by Activity Level") #Sets the title of the plot.
plt.xlabel("Activity Level") #Labels the axes.
plt.ylabel("Calories Burned") #Labels the axes.
plt.show() #Displays the plot.

# Scatter Plot: Actual vs Predicted Calories Burned
plt.figure(figsize=(8, 6)) 
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7) #Creates a scatter plot comparing actual and predicted calories burned.
plt.title("Actual vs Predicted Calories Burned") 
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.axline([0, 0], [1, 1], color='red', linestyle='--', label='Perfect Prediction')  #Adds a reference line for perfect prediction.
plt.legend() #Adds a legend.
plt.show()

# Regression Model Evaluation: Residual Plot
residuals = y_test - y_pred #Calculates residuals (difference between actual and predicted values).
plt.figure(figsize=(8, 6)) 
sns.histplot(residuals, kde=True, bins=30, color='blue') #Creates a histogram of residuals.
plt.title("Residuals Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.axvline(0, color='red', linestyle='--', label='Zero Residual') #Adds a vertical line at zero residual.
plt.legend()
plt.show()

# Implications for Software Engineering Decision-Making
print("\nImplications:")
print("1. Regression analysis helps predict user calorie burn based on app usage.")
print("2. Clustering identifies distinct user groups for targeted feature development.")
print("3. Insights from these analyses can guide personalized app experiences, improve user retention, and prioritize development efforts.")
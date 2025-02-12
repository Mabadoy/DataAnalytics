import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Sample_Data_for_Activity.csv")
print("Print top 5 records:")
print(df.head())

print("Data Information")
print(df.info())

sns.displot(df['Normal_Distribution'], kde=True, bins=50)
sns.jointplot(x='Normal_Distribution', y='Uniform_Distribution', data=df, kind='scatter')

plt.show()
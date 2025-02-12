import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
titanic = sns.load_dataset('titanic')
print(titanic.head())

sns.jointplot(x='fare',y='age',data=titanic)
plt.show()

sns.displot(titanic['fare'],bins=30,kde=False,color='blue')
plt.show()

sns.boxplot(x='class',y='age',data=titanic,palette='Set3',hue='class')
plt.show()

sns.swarmplot(x='class',y='age',data=titanic,palette='pastel6',hue='class')
plt.show()

sns.countplot(x='sex',data=titanic)
plt.show()

titanic_corr = titanic[["survived", "pclass", "age", "sibsp", "parch", "fare"]].corr()
sns.heatmap(titanic_corr,cmap='coolwarm', annot=True)
plt.title('titanic.corr()')
plt.show()

g = sns.FacetGrid(data=titanic,col='sex')
g.map(plt.hist,'age')
plt.show()
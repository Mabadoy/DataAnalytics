import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 1, 5]
y = [10, 20, 25, 30, 40, 10]
plt.plot(x, y, marker='o', color='red')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Basic Line Plot')
plt.show()

categories = ['Cloud Security', 'Data Analytics', 'BlockChain', 'ITS']
values = [88, 97, 82, 91]
plt.bar(categories, values, color='skyblue')
plt.xlabel('Subjects')
plt.ylabel('Grades')
plt.title('Yoobee')
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Data
study_hours = [1, 2, 3, 4, 5, 6, 7 ,8, 9, 10]
test_scores = [50, 55, 65, 70, 72, 78, 85, 88, 92, 95]

# Create a scatter plot
plt.figure(figsize=(8, 6))  # Set the figure size
plt.scatter(study_hours, test_scores, color='blue', marker='o', label='Data Points')

# Add labels and title
plt.title('Scatter Plot: Study Hours vs Test Scores', fontsize=14)
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Test Scores', fontsize=12)

# Add a trendline
z = np.polyfit(study_hours, test_scores, 1)  # Fit a linear trendline
p = np.poly1d(z)
plt.plot(study_hours, p(study_hours), color='red', linestyle='--', label='Trendline')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Observations
# Trend: The scatter plot shows a positive correlation between study hours and test scores. 
# As the number of study hours increases, test scores also tend to increase.

# Trendline: Adding a linear trendline (red dashed line) confirms the positive relationship. 
# The data suggests that students who study more hours generally achieve higher test scores.

# This visualization highlights the importance of study time in improving test performance!
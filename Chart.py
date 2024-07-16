import matplotlib.pyplot as plt
import numpy as np

# Models
models = ['Perceptron', 'SVM', 'Decision Tree', 'Neural Network', 'Logistic Regression']

# Metrics values for each model
accuracy_values = [0.5163, 0.99116, 0.99686, 0.9933, 0.66916]
precision_values = [0.6972, 0.9912, 0.9969, 0.9930, 0.6571]
recall_values = [0.5163, 0.9913, 0.99686, 0.9931, 0.6692]
f1_score_values = [0.4570, 0.9913, 0.99676, 0.9933, 0.6585]

# Bar width
barWidth = 0.2

# Set positions for bar groups
r1 = np.arange(len(models))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Plotting the metrics for each model
plt.bar(r1, accuracy_values, width=barWidth, label='Accuracy', color='blue')
plt.bar(r2, precision_values, width=barWidth, label='Precision', color='green')
plt.bar(r3, recall_values, width=barWidth, label='Recall', color='orange')
plt.bar(r4, f1_score_values, width=barWidth, label='F1 Score', color='red')

# Adding labels
plt.xlabel('Mô hình', fontweight='bold')
plt.xticks([r + 1.5 * barWidth for r in range(len(models))], models)
plt.title('Biểu đồ so sánh các độ đo các mô hình')
plt.legend()

# Display the bar chart
plt.show()

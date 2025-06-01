import numpy as np
import matplotlib.pyplot as plt

# Load y data
y = np.load('y.npy')

# Get the number of samples per class
unique, counts = np.unique(y, return_counts=True)

# Plot bar chart
plt.figure(figsize=(6, 4))
plt.bar(unique, counts, color=['green', 'red'])
plt.xticks(unique, ['Normal', 'Abnormal'])  # Adjust based on actual labels
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Distribution of Labels (y)')
plt.grid(axis='y')

# Save the plot as image
plt.savefig('label_distribution.png')

# Show the plot
plt.show()
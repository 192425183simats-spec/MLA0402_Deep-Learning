import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Actual values
actual = np.array([
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Dog',
    'Not Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

# Predicted values
predicted = np.array([
    'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog',
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

# Create confusion matrix
conf_matrix = confusion_matrix(actual, predicted, labels=['Dog', 'Not Dog'])

# Display confusion matrix
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='g',
    xticklabels=['Dog', 'Not Dog'],
    yticklabels=['Dog', 'Not Dog'],
    cmap='RdPu'
)

plt.ylabel("Prediction", fontsize=16)
plt.xlabel("Actual", fontsize=16)
plt.title("Confusion Matrix", fontsize=20)

plt.show()

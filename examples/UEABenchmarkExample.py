import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from aeon.datasets import load_classification
from src.PHeatPruner import PHeatPruner
import shap

# Load the dataset
dataset = "NATOPS"  # or any other dataset in the UEA Archive
X_train, y_train = load_classification(dataset, split="train")
X_test, y_test = load_classification(dataset, split="test")

# Encode labels as integer indices
y_train = [np.where(np.unique(y_train) == label)[0][0] for label in y_train]
y_train_df = pd.DataFrame(y_train)
y_test = [np.where(np.unique(y_test) == label)[0][0] for label in y_test]
y_test_df = pd.DataFrame(y_test)

# Prune the dataset using PHeatPruner
pruned_X_train, pruned_X_test = PHeatPruner(X_train, X_test)

# Train a RandomForestClassifier on the pruned data
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(pruned_X_train, y_train_df)
predictions = rf_clf.predict(pruned_X_test)

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test_df, predictions))
ConfusionMatrixDisplay.from_predictions(y_test_df, predictions)
plt.title('Confusion Matrix')
plt.show()

# Print the classification report
print("Classification Report:")
print(classification_report(y_test_df, predictions))

# Explain the model using SHAP
explainer = shap.TreeExplainer(rf_clf)
explanation = explainer(pruned_X_test)
shap.plots.beeswarm(explanation[:, :, 0], max_display=40)
plt.show()

# Re-prune the data with sheafification
pruned_X_train_sheaf, pruned_X_test_sheaf = PHeatPruner(X_train, X_test, sheafification=True)

# Retrain the model on the sheafified data
rf_clf.fit(pruned_X_train_sheaf, y_train_df)
predictions_sheaf = rf_clf.predict(pruned_X_test_sheaf)

# Explain the sheafified data model using SHAP
explanation_sheaf = explainer(pruned_X_test_sheaf)
shap.plots.beeswarm(explanation_sheaf[:, :, 0], max_display=40)
plt.show()

# Display the confusion matrix for the sheafified data
print("Confusion Matrix (Sheafified Data):")
print(confusion_matrix(y_test_df, predictions_sheaf))
ConfusionMatrixDisplay.from_predictions(y_test_df, predictions_sheaf)
plt.title('Confusion Matrix (Sheafified Data)')
plt.show()

# Print the classification report for the sheafified data
print("Classification Report (Sheafified Data):")
print(classification_report(y_test_df, predictions_sheaf))

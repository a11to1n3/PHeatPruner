# PHeatPruner

`PHeatPruner` is a Python function designed to prune variables in both multivariate time-series and tabular datasets using persistent homology analysis. It is distributed as part of the **LimeSoDa** library for topological data analysis. The function also offers an optional sheafification process to enhance the feature set, making it useful for dimensionality reduction while maintaining the essential structure for machine learning tasks.

## Installation

To use `PHeatPruner`, you need to install the following dependencies.  The
pruner itself is bundled in the `LimeSoDa` package, so installing that package
will pull in the code found in this repository:

- `numpy`
- `pandas`
- `tqdm`
- `gudhi`
- `matplotlib`
- `scikit-learn`
- `shap`
- `aeon`
- `tabpfn`
- `limesoda`

You can install these using pip:

```bash
pip install numpy pandas tqdm gudhi matplotlib scikit-learn shap aeon tabpfn limesoda
```

## Usage
Here’s an example (also in [here](examples/UEABenchmarkExample.py)) of how to use PHeatPruner with a dataset from the UEA Archive:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabpfn import TabPFNClassifier
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

# Train a TabPFNClassifier on the pruned data
rf_clf = TabPFNClassifier(device="cpu")
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
```

For tabular datasets simply pass your ``numpy`` or ``pandas`` arrays to ``PHeatPruner``::

```python
import pandas as pd
from src.PHeatPruner import PHeatPruner

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

pruned_train, pruned_test = PHeatPruner(df_train.values, df_test.values)

# Fit a quick TabPFNRegressor on the pruned data
from tabpfn import TabPFNRegressor
reg = TabPFNRegressor(device="cpu")
reg.fit(pruned_train, y_train)
predictions = reg.predict(pruned_test)
```

## Comparison with other feature selection methods

PHeatPruner focuses on structural relationships among features using persistent
homology.  In practice it complements traditional filter and wrapper methods.

* **Correlation/Mutual Information Filters** – These quickly remove highly
  redundant variables but may miss nonlinear or higher-order interactions.
  PHeatPruner detects such interactions through connected simplices.
* **Recursive Feature Elimination (RFE)** – RFE iteratively trains a model and
  removes the least important features.  When applied to UEA time-series
  datasets, PHeatPruner can provide a smaller search space for RFE and reduce
  runtime.
* **SHAP-based Importance** – SHAP values explain model predictions but require
  a trained model.  PHeatPruner offers a model-agnostic pre-filtering step
  suitable for both UEA Archive tasks and generic LimeSoDa applications.
* **L1-penalized Models** – Logistic regression with an L1 penalty or Lasso
  regression performs embedded feature selection.  PHeatPruner can further
  reduce the feature count before these models, simplifying the optimization.

Other state-of-the-art packages such as gradient boosting frameworks or TabNet
also provide built-in feature importance measures.  PHeatPruner acts as a
lightweight pre-processing step that can be used ahead of these approaches.

In practice, running PHeatPruner prior to these methods decreases the number of
variables they must examine.  This often speeds up downstream feature selection
without reducing predictive performance.

### Benchmarking

The `examples/LimeSoDaBenchmark.py` script compares PHeatPruner with
correlation filtering, RFE, a SHAP-based selector, and a logistic-regression
L1 selector on a dataset loaded from the `limesoda` library.  It trains a
`TabPFNClassifier` on the features returned by each method and prints their
respective accuracies.

```bash
python examples/LimeSoDaBenchmark.py
```

## Note
- Persistent Homology: The pruning threshold is selected from the longest persistent feature in the data rather than relying on heuristics.
- Sheafification: An optional process that enhances the feature set by considering higher-order interactions among the variables.

## LICENSE
This project is licensed under the Apache License - see the [LICENSE](./LICENSE) file for details.

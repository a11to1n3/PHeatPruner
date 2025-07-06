"""Benchmarking pipeline comparing PHeatPruner with other feature selection methods on a LimeSoDa dataset."""

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap

from limesoda.datasets import load_classification
from src.PHeatPruner import PHeatPruner


def _flatten_dataset(X):
    """Convert a 3D time series array to a 2D tabular DataFrame."""
    X = np.asarray(X)
    if X.ndim == 3:
        n_samples, n_vars, n_time = X.shape
        data = {
            f"Var{i}_T{j}": X[:, i, j]
            for i in range(n_vars)
            for j in range(n_time)
        }
        return pd.DataFrame(data)
    return pd.DataFrame(X)


def corr_filter(df, threshold=0.9):
    """Simple correlation filter removing highly correlated columns."""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > threshold)]
    return df.drop(drop_cols, axis=1)


def rfe_selection(X_train, y_train, X_test, n_feats):
    """Recursive feature elimination using a random forest."""
    estimator = RandomForestClassifier(n_estimators=100, random_state=0)
    selector = RFE(estimator, n_features_to_select=n_feats, step=0.1)
    selector.fit(X_train, y_train)
    return selector.transform(X_train), selector.transform(X_test)


def shap_selection(X_train, y_train, X_test, top_k):
    """Select features with the highest mean absolute SHAP values."""
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_train)
    shap_mean = np.mean(np.abs(shap_values), axis=1).mean(axis=0)
    idx = np.argsort(shap_mean)[-top_k:]
    return X_train[:, idx], X_test[:, idx]


def l1_logreg_selection(X_train, y_train, X_test, max_feats):
    """Select features using logistic regression with L1 penalty."""
    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=200)
    selector = SelectFromModel(model, max_features=max_feats)
    selector.fit(X_train, y_train)
    return selector.transform(X_train), selector.transform(X_test)


def run_benchmark(dataset="ArticularyWordRecognition"):
    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")

    uniq = np.unique(y_train)
    y_train = np.array([np.where(uniq == y)[0][0] for y in y_train])
    y_test = np.array([np.where(uniq == y)[0][0] for y in y_test])

    # Baseline TabPFN on flattened data
    X_train_df = _flatten_dataset(X_train)
    X_test_df = _flatten_dataset(X_test)
    clf = TabPFNClassifier(device="cpu")
    clf.fit(X_train_df, y_train)
    pred = clf.predict(X_test_df)
    baseline_acc = accuracy_score(y_test, pred)

    # Correlation filter
    cf_X_train = corr_filter(X_train_df)
    cf_X_test = X_test_df[cf_X_train.columns]
    clf.fit(cf_X_train, y_train)
    pred = clf.predict(cf_X_test)
    corr_acc = accuracy_score(y_test, pred)

    # RFE
    rfe_train, rfe_test = rfe_selection(X_train_df.values, y_train, X_test_df.values, n_feats=cf_X_train.shape[1])
    clf.fit(rfe_train, y_train)
    pred = clf.predict(rfe_test)
    rfe_acc = accuracy_score(y_test, pred)

    # SHAP selection
    shap_train, shap_test = shap_selection(X_train_df.values, y_train, X_test_df.values, top_k=cf_X_train.shape[1])
    clf.fit(shap_train, y_train)
    pred = clf.predict(shap_test)
    shap_acc = accuracy_score(y_test, pred)

    # Logistic regression with L1 penalty
    l1_train, l1_test = l1_logreg_selection(X_train_df.values, y_train, X_test_df.values, max_feats=cf_X_train.shape[1])
    clf.fit(l1_train, y_train)
    pred = clf.predict(l1_test)
    l1_acc = accuracy_score(y_test, pred)

    # PHeatPruner
    pruned_train, pruned_test = PHeatPruner(X_train, X_test)
    clf.fit(pruned_train, y_train)
    pred = clf.predict(pruned_test)
    pruner_acc = accuracy_score(y_test, pred)

    print("Baseline accuracy:", baseline_acc)
    print("Correlation filter accuracy:", corr_acc)
    print("RFE accuracy:", rfe_acc)
    print("SHAP accuracy:", shap_acc)
    print("L1 logistic regression accuracy:", l1_acc)
    print("PHeatPruner accuracy:", pruner_acc)


if __name__ == "__main__":
    run_benchmark()

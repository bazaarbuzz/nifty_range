# utils/evaluation.py
import logging
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from config import constants

def calculate_regression_metrics(y_true, y_pred):
    """Calculates standard regression metrics."""
    metrics = {}
    try:
        metrics[constants.METRICS_RMSE] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[constants.METRICS_MAE] = mean_absolute_error(y_true, y_pred)
        metrics[constants.METRICS_R2] = r2_score(y_true, y_pred)
    except Exception as e:
        print(f"Error calculating regression metrics: {e}")
    return metrics

def calculate_classification_metrics(y_true, y_pred, y_proba=None):
    """Calculates standard classification metrics."""
    metrics = {}
    try:
        metrics[constants.METRICS_ACCURACY] = accuracy_score(y_true, y_pred)
        # Use weighted average for multi-class, binary otherwise (adjust if needed)
        avg_method = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
        metrics[constants.METRICS_PRECISION] = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
        metrics[constants.METRICS_RECALL] = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
        metrics[constants.METRICS_F1] = f1_score(y_true, y_pred, average=avg_method, zero_division=0)
        if y_proba is not None:
            try:
                 # Handle multi-class ROC AUC if needed (requires OvR/OvO strategy)
                 if len(y_proba.shape) > 1 and y_proba.shape[1] > 2: # Multi-class probabilities
                      metrics[constants.METRICS_ROC_AUC] = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
                 elif len(y_proba.shape) > 1 and y_proba.shape[1] == 2: # Binary probabilities
                      metrics[constants.METRICS_ROC_AUC] = roc_auc_score(y_true, y_proba[:, 1])
                 else: # Possibly just scores, not probabilities
                     logging.warning("Cannot calculate ROC AUC: Invalid probability shape.")
            except ValueError as roc_e:
                 logging.warning(f"Could not calculate ROC AUC: {roc_e}")

        # Confusion Matrix is usually plotted, not stored as a single metric value
        # cm = confusion_matrix(y_true, y_pred)

    except Exception as e:
        print(f"Error calculating classification metrics: {e}")
    return metrics


def calculate_clustering_metrics(X, labels):
    """Calculates standard clustering metrics."""
    metrics = {}
    n_labels = len(set(labels))
    n_samples = len(X)

    if n_labels < 2 or n_labels >= n_samples:
         logging.warning(f"Cannot calculate clustering metrics with {n_labels} labels for {n_samples} samples.")
         return metrics

    try:
        metrics[constants.METRICS_SILHOUETTE] = silhouette_score(X, labels)
        metrics[constants.METRICS_CALINSKI] = calinski_harabasz_score(X, labels)
        metrics[constants.METRICS_DAVIES] = davies_bouldin_score(X, labels)
    except Exception as e:
        print(f"Error calculating clustering metrics: {e}")
    return metrics
# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA # Needed for cluster plotting if PCA used
try:
    import umap # Needed for cluster plotting if UMAP used
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from config.settings import PLOT_OUTPUT_DIR, FIG_SIZE, CORR_MATRIX_FIG_SIZE, FEATURE_IMPORTANCE_TOP_N

sns.set(style="whitegrid")

def save_plot(fig, filename, directory=PLOT_OUTPUT_DIR):
    """Saves the matplotlib figure to a file."""
    try:
        path = os.path.join(directory, filename)
        fig.savefig(path, bbox_inches='tight')
        logging.info(f"Plot saved to {path}")
        plt.close(fig) # Close the figure to free memory
    except Exception as e:
        logging.error(f"Error saving plot {filename}: {e}")

def plot_correlation_matrix(df, title="Correlation Matrix", filename="correlation_matrix.png"):
    """Plots and saves the correlation matrix of a DataFrame."""
    if df.empty:
        logging.warning("Cannot plot correlation matrix for empty DataFrame.")
        return
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
         logging.warning(f"Not enough numeric columns ({numeric_df.shape[1]}) to plot correlation matrix.")
         return

    fig, ax = plt.subplots(figsize=CORR_MATRIX_FIG_SIZE)
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, linewidths=.5)
    ax.set_title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_plot(fig, filename)

def plot_feature_importance(importance_df, title="Feature Importance", filename="feature_importance.png"):
    """Plots and saves feature importances."""
    if importance_df.empty:
        logging.warning("Cannot plot empty feature importance DataFrame.")
        return

    top_n = min(FEATURE_IMPORTANCE_TOP_N, len(importance_df))
    importance_df = importance_df.nlargest(top_n, 'Importance')

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    save_plot(fig, filename)

def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual", filename="predictions_vs_actual.png"):
    """Plots predictions against actual values."""
    if len(y_true) != len(y_pred):
        logging.error("Length mismatch between y_true and y_pred.")
        return
    if len(y_true) == 0:
        logging.warning("Cannot plot empty predictions.")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', s=80)
    # Add identity line (y=x)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label='Ideal Fit (y=x)')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save_plot(fig, filename)

def plot_distribution(series, title="Distribution Plot", filename="distribution.png"):
    """Plots the distribution of a pandas Series."""
    if series.empty:
        logging.warning("Cannot plot distribution for empty Series.")
        return
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.histplot(series, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    save_plot(fig, filename)

def plot_time_series(series, title="Time Series Plot", filename="time_series.png"):
    """Plots a time series."""
    if series.empty:
        logging.warning("Cannot plot empty time series.")
        return
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    series.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(series.name)
    plt.tight_layout()
    save_plot(fig, filename)

def plot_seasonality(series, period='M', title_suffix="Monthly Seasonality", filename_suffix="monthly_seasonality.png"):
    """Plots seasonality by aggregating data over a period (e.g., 'M' for month, 'Q' for quarter)."""
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        logging.warning("Cannot plot seasonality for non-datetime indexed or empty Series.")
        return

    if period == 'M':
        grouper = series.index.month
        group_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    elif period == 'Q':
        grouper = series.index.quarter
        group_names = ['Q1', 'Q2', 'Q3', 'Q4']
    elif period == 'W': # Day of week
        grouper = series.index.dayofweek
        group_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] # Adjust if market closed on weekends
    else:
        logging.error(f"Unsupported seasonality period: {period}. Use 'M', 'Q', or 'W'.")
        return

    seasonal_data = series.groupby(grouper).mean()
    if len(seasonal_data) != len(group_names): # Handle cases where not all groups are present
        seasonal_data = seasonal_data.reindex(range(len(group_names)), fill_value=np.nan)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    seasonal_data.plot(kind='bar', ax=ax)
    ax.set_title(f"{series.name} {title_suffix}")
    ax.set_xlabel("Period")
    ax.set_ylabel(f"Average {series.name}")
    ax.set_xticklabels(group_names, rotation=45)
    plt.tight_layout()
    save_plot(fig, f"{series.name.lower().replace(' ', '_')}_{filename_suffix}")

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix", filename="confusion_matrix.png"):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes) # Ensure labels match classes order
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)
    plt.tight_layout()
    save_plot(fig, filename)


def plot_actual_vs_predicted_timeseries(y_true, y_pred, title="Actual vs Predicted Time Series", filename="actual_vs_predicted_ts.png"):
     """Plots actual and predicted values over time (assumes y_true and y_pred have datetime index)."""
     if not isinstance(y_true.index, pd.DatetimeIndex) or not isinstance(y_pred.index, pd.DatetimeIndex):
         logging.warning("Cannot plot time series: Indices are not DatetimeIndex.")
         return
     if not y_true.index.equals(y_pred.index):
          logging.warning("Indices of y_true and y_pred do not match. Aligning...")
          # Attempt simple alignment, might fail for complex cases
          y_true_aligned, y_pred_aligned = y_true.align(y_pred, join='inner')
          if y_true_aligned.empty:
               logging.error("Could not align time series indices.")
               return
          y_true, y_pred = y_true_aligned, y_pred_aligned


     fig, ax = plt.subplots(figsize=(15, 7))
     ax.plot(y_true.index, y_true, label='Actual', marker='.', linestyle='-')
     ax.plot(y_pred.index, y_pred, label='Predicted', marker='x', linestyle='--')
     ax.set_xlabel("Date")
     ax.set_ylabel("Value")
     ax.set_title(title)
     ax.legend()
     ax.grid(True)
     plt.xticks(rotation=45)
     plt.tight_layout()
     save_plot(fig, filename)


def plot_clusters(X_reduced, labels, method_name="PCA", title="Cluster Visualization", filename="clusters.png"):
    """Plots clustered data reduced to 2D or 3D."""
    dims = X_reduced.shape[1]
    if dims not in [2, 3]:
        logging.error(f"Cannot plot clusters: Data must be reduced to 2 or 3 dimensions, got {dims}.")
        return

    unique_labels = sorted(list(set(labels)))
    # Handle potential noise label from DBSCAN (-1)
    is_noise = -1 in unique_labels
    n_clusters_ = len(unique_labels) - (1 if is_noise else 0)

    fig = plt.figure(figsize=(10, 8))

    if dims == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel(f'{method_name} Dimension 1')
        ax.set_ylabel(f'{method_name} Dimension 2')
    else: # dims == 3
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels, cmap='viridis', s=50, alpha=0.6)
        ax.set_xlabel(f'{method_name} Dimension 1')
        ax.set_ylabel(f'{method_name} Dimension 2')
        ax.set_zlabel(f'{method_name} Dimension 3')

    ax.set_title(f'{title} ({n_clusters_} clusters found via {method_name})')

    # Create legend - handle noise label separately if present
    clean_labels = [l for l in unique_labels if l != -1]
    clean_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {l}',
                               markerfacecolor=scatter.cmap(scatter.norm(l)), markersize=10) for l in clean_labels]
    if is_noise:
         noise_handle = plt.Line2D([0], [0], marker='o', color='w', label='Noise (-1)',
                                markerfacecolor=scatter.cmap(scatter.norm(-1)), markersize=10)
         clean_handles.append(noise_handle)

    ax.legend(handles=clean_handles, title="Clusters")
    plt.tight_layout()
    save_plot(fig, filename)
# Add more plotting functions as needed (e.g., box plots, violin plots, etc.)
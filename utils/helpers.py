"""
Helper functions for Triangle ML
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Tuple, Optional


def load_custom_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[list], Optional[list]]:
    """
    Load a custom dataset from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names (if available)
        target_names: List of target names (if available)
    """
    try:
        df = pd.read_csv(file_path)

        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        feature_names = df.columns[:-1].tolist()
        target_names = None

        # Try to infer target names for classification
        if len(np.unique(y)) <= 20:  # Reasonable number of classes
            target_names = [f'class_{i}' for i in np.unique(y)]

        return X, y, feature_names, target_names

    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[list] = None) -> go.Figure:
    """
    Create an interactive confusion matrix plot.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Plotly figure object
    """
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=500,
        height=500
    )

    return fig


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> go.Figure:
    """
    Create ROC curve for multi-class classification.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_classes: Number of classes

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random',
        showlegend=False
    ))

    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc_score = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=3)
        ))
    else:
        # Multi-class classification (One-vs-Rest)
        for i in range(n_classes):
            y_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
            auc_score = auc(fpr, tpr)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'Class {i} (AUC = {auc_score:.3f})'
            ))

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=600,
        height=500,
        showlegend=True
    )

    return fig


def get_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive model metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary containing various metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, roc_auc_score
    )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }

    # Add AUC if probabilities are provided
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = None

    return metrics


def validate_dataset(X: np.ndarray, y: np.ndarray) -> bool:
    """
    Validate dataset for common issues.

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        True if dataset is valid, raises ValueError otherwise
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must match")

    if X.shape[0] == 0:
        raise ValueError("Dataset is empty")

    if np.isnan(X).any():
        raise ValueError("Dataset contains NaN values in features")

    if np.isnan(y).any():
        raise ValueError("Dataset contains NaN values in target")

    return True


def preprocess_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Preprocess data for training.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility

    Returns:
        Preprocessed X_train, X_test, y_train, y_test, scaler
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Validate dataset
    validate_dataset(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained model object
        filepath: Path to save the model
    """
    import joblib
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model object
    """
    import joblib
    return joblib.load(filepath)
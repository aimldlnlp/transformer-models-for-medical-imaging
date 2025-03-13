import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def calculate_metrics(y_true, y_pred, model_name):
    """Computes and prints evaluation metrics."""
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred_classes))

    print(f"\nConfusion Matrix for {model_name}:\n")
    print(confusion_matrix(y_true, y_pred_classes))
from sklearn.metrics import *
from scipy.special import softmax
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to probabilities to match Cross Entropy Loss
    probabilities = softmax(logits, axis=1)

    # Positive class probability
    y_score = probabilities[:, 1]

    # Predicted labels
    y_pred = np.argmax(probabilities, axis=1)

    # Accuracy
    acc = accuracy_score(labels, y_pred)

    # Macro Precision / Recall / F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        y_pred,
        average='macro',
        zero_division=0
    )

    # PR-AUC
    auc_pr = average_precision_score(labels, y_score)

    return {
        "accuracy": acc,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "auc_pr": auc_pr
    }

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def compute_metrics(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def optimize_thresholds(y_true, y_probs):
    thresholds = []

    for i in range(y_true.shape[1]):
        best_thresh = 0.5
        best_f1 = 0

        for t in np.arange(0.1, 0.9, 0.05):
            preds = (y_probs[:, i] > t).astype(int)
            score = f1_score(y_true[:, i], preds)

            if score > best_f1:
                best_f1 = score
                best_thresh = t

        thresholds.append(best_thresh)

    return thresholds

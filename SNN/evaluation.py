import numpy as np
import pandas as pd

def classification_report(y_true, y_pred):
    #changing predictions to 0/1:
    y_pred = (y_pred == y_pred.max(axis=1, keepdims=True))*1

    #recall:
    label_counts = y_true.sum(axis=0)
    #changing zeros to ones in denominator, since then 0 in numerator anyway
    label_counts[label_counts == 0]=1
    recall = ((y_pred * y_true).sum(axis=0)/label_counts)

    #precision:
    prediction_counts = y_pred.sum(axis=0)
    #changing zeros to ones in denominator, since then 0 in numerator anyway
    prediction_counts[prediction_counts == 0]=1
    precision = ((y_pred * y_true).sum(axis=0)/prediction_counts)

    f_scores = (2*(precision*recall)/(precision+recall))
    f_scores[np.isnan(f_scores)] = 0

    classification = {
     'class': np.arange(len(recall)),
     'recall': recall,
     'precision': precision,
     'f_scores': f_scores,
     'support': y_true.sum(axis=0),
    }

    return pd.DataFrame(classification)

def confusion_matrix(y_true, y_pred):
    #changing predictions to 0/1:
    y_pred = (y_pred == y_pred.max(axis=1, keepdims=True))*1
    supports = y_true.sum(axis=0, keepdims=True)
    return (y_pred.T @ y_true)/supports

import numpy as np
import math

def accuracy(confusion_matrix) -> float:
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)

def precision(confusion_matrix) -> float:
    return confusion_matrix[0][0] / np.sum(confusion_matrix[:, 0])

def recall(confusion_matrix) -> float:
    return confusion_matrix[0][0] / np.sum(confusion_matrix[0])

def f1_score(confusion_matrix) -> float:
    p = precision(confusion_matrix)
    r = recall(confusion_matrix)
    return 2 * p * r / (p + r)
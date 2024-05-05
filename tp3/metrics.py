import numpy as np
import math

def accuracy(confusion_matrix) -> float:
    sum = np.sum(confusion_matrix)
    if sum == 0:
        return 0
    return np.trace(confusion_matrix) / sum

def precision(confusion_matrix) -> float:
    sum =  np.sum(confusion_matrix[:, 0])
    if sum == 0:
        return 0
    return confusion_matrix[0][0] / sum

def recall(confusion_matrix) -> float:
    sum = np.sum(confusion_matrix[0])
    if sum == 0:
        return 0
    return confusion_matrix[0][0] / sum

def f1_score(confusion_matrix) -> float:
    p = precision(confusion_matrix)
    r = recall(confusion_matrix)
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)
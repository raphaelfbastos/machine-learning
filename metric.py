import numpy as np

def R2(a:np.ndarray, y:np.ndarray) -> float:
    return 1 - np.sum((y - a) ** 2) / np.sum((y - np.mean(y)) ** 2)

def adjusted_R2(a:np.ndarray, y:np.ndarray, n:int, p:int) -> float:
    return 1 - (1 - R2(a, y)) * (n - 1) / (n - p - 1)
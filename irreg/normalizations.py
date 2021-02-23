import numpy as np


def min_max_scaler(data: np.ndarray, min: float = 0, max: float = 1):
    data = (data - data.min()) / (data.max() - data.min())
    data = data * (max - min) + min
    return data


def standard_scaler(data: np.ndarray):
    data = (data - data.mean()) / data.std()
    return data

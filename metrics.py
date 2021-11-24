import torch
import numpy as np


def smape_metrics(y_true, y_hat):
    # note: final_smape = smape * 100
    error = np.mean(np.square(y_true - y_hat))
    smape = np.mean(2 * np.abs(y_true - y_hat) / (np.abs(y_true) + np.abs(y_hat)))
    return smape, error



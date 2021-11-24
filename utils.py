import numpy
import torch.utils.data
import numpy as np


def read_target(X, window, pred_num, mode):

    result = []
    for i in range(len(X)):
        if mode == 'train':
            for j in range(window, (X[i].shape[0]-pred_num)):
                result.append(X[i][j:j+pred_num])
        else:
            j = X[i].shape[0] - pred_num
            result.append(X[i][j:j+pred_num])

    return np.array(result, dtype=np.double)


def read_term_pred_term(X, terms, pred_terms):
    # X: sample_num * (uncertainty)

    result, labels = [], []
    for b in range(X.shape[0]):
        if X[b].shape[0] > terms + pred_terms:
            for i in range(0, X[b].shape[0]-terms-pred_terms):
                result.append(X[b][i:i+terms+pred_terms])
                labels.append(b)

        else:
            print('over window')
            exit()
    result, labels = np.array(result), np.array(labels)
    result = result[np.newaxis, :]
    return result, labels


def fill_pre_nan(max_seq_length, train):
    nan_len_train = np.zeros(len(train))
    result_train = []
    for i in range(len(train)):
        if train[i].shape[0] < max_seq_length:
            tmp = list(train[i]) + [np.nan] * (max_seq_length - train[i].shape[0])
            nan_len_train[i] = max_seq_length - train[i].shape[0]
            result_train.append(tmp)
        else:
            result_train.append(list(train[i]))
    result_train = np.array(result_train)
    result_train = result_train[np.newaxis, :]
    result_train = np.swapaxes(result_train, 1, 0)
    return result_train, nan_len_train


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        data = self.dataset[index] # ori_series
        return data


class LabelledDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]
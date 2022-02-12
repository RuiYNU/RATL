import numpy as np
import datetime
import sys
import argparse

import torch
import torch.utils.data
import argparse
from torch import nn
from torch.optim import SGD
from utils import fill_pre_nan, Dataset
from encoder_models import Timeseries_Encoder as Model
import torch.nn.functional as F
import pickle as pkl
from utils import read_target
from metrics import smape_metrics


def get_data(fpath_train='NN5/daily.train',
             fpath_test='NN5/daily.test',
             batch_size=30,
             if_nan=False):
    train = np.load(fpath_train + '.npy', allow_pickle=True)
    test = np.load(fpath_test + '.npy', allow_pickle=True)

    max_seq_length = 1
    for i in range(len(train)):
        max_seq_length = max(train[i].shape[0], max_seq_length)

    for i in range(len(test)):
        max_seq_length = max(test[i].shape[0], max_seq_length)

    if if_nan:
        train_nan, nan_len_train = fill_pre_nan(max_seq_length, train)
        test_nan, nan_len_test = fill_pre_nan(max_seq_length, test)
    else:
        train_nan = train[np.newaxis, :]
        train_nan = np.swapaxes(train_nan, 1, 0)
        test_nan = test[np.newaxis, :]
        test_nan = np.swapaxes(test_nan, 1, 0)

    print('TRAIN:', train_nan.shape, test_nan.shape)

    train_torch_dataset = Dataset(train_nan)
    train_generator = torch.utils.data.DataLoader(
        train_torch_dataset, batch_size=batch_size, shuffle=True
    )

    test_torch_dataset = Dataset(test_nan)
    test_generator = torch.utils.data.DataLoader(
        test_torch_dataset, batch_size=batch_size, shuffle=True
    )
    return train_generator, test_generator, train_nan, test_nan, train, test


def save_linear(linear, linear_file="models_save/m3_quarterly"):
    torch.save(
        linear.state_dict(),
        linear_file + '_linear.pth'
    )


def encoder_linear_train(train, train_, test_,
                         encode_pred_num, encode_window,
                         window_size, test_pred_num, epochs=400,
                         training=True, compute_representation=True,
                         storage_train='dataset_save/nn5_train.npy',
                         storage_test='dataset_save/nn5_test.npy',
                         linear_file='models_save/nn5',
                         model_file='../models_save/nn5'):
    cuda, gpu = False, 0

    if torch.cuda.is_available():
        print("Using CUDA...")
        cuda = True

    parameters = {'compared_length': None, 'neg_samples': 10,
                  'batch_size': 16, 'train_steps': 300, 'lr': 0.001,
                  'n_cluster': 5, 'in_channels': 1, 'channels': 30,
                  'depth': 10, 'reduced_size': 80, 'kernel_size': 3,
                  'out_channels': 160, 'cuda': cuda, 'gpu': gpu, 'module': 'gru',
                  'enc_hidden': 1, 'ar_hidden': 24, 'prediction_step': 6}

    encoder = Model.CausalCNN_Time()
    encoder.set_params(**parameters)
    #encoder.batch_size = train_batch_size

    if training:
        encoder.fit_encoder(train)
        print('Fit_encoder end')
        encoder.save_encoder(model_file)
    else:
        encoder.load_encoder(model_file)

    #train_, test_ = train.reshape((train.shape[0], -1)), test.reshape((test.shape[0], -1))

    if compute_representation:
        train_features = encoder.encode_window_new(train_, encode_window, encode_pred_num, 'train')
        with open(storage_train, 'wb') as f:
            pkl.dump([train_features], f)
#####################################################################################
        test_features = encoder.encode_window_new(test_,  encode_window, encode_pred_num, 'test')
#####################################################################################
        with open(storage_test, 'wb') as f:
            pkl.dump([test_features], f)
    else:
        with open(storage_train, 'rb') as f:
            train_features = pkl.load(f)

        with open(storage_test, 'rb') as f:
            test_features = pkl.load(f)

    train_features, test_features = np.array(train_features).reshape((-1, 160)), np.array(test_features).reshape((-1, 160))

    train_features = torch.from_numpy(train_features)
    test_features = torch.from_numpy(test_features)

    if torch.cuda.is_available():
        train_features = train_features.cuda(gpu)
        test_features = test_features.cuda(gpu)

    train_target = read_target(train_, window_size, test_pred_num, mode='train')
    test_target = read_target(test_, window_size, test_pred_num, mode='test')

    train_target = torch.from_numpy(train_target)
    test_target = torch.from_numpy(test_target)
    if torch.cuda.is_available():
        train_target = train_target.cuda(gpu)
        test_target = test_target.cuda(gpu)

    print('train_target.shape:', train_target.shape)
    print('test_target.shape:', test_target.shape)

    # Predictor
    regressor = nn.Linear(160, test_pred_num)
    regressor.double()
    if torch.cuda.is_available():
        regressor.cuda(gpu)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.01)

    best_smape = 999999.0
    best_epoch = 0
    for i in range(epochs):
        y_pred_train = regressor(train_features)

        l = loss(y_pred_train, train_target)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 50 == 0:
            print('epoch ', i, ', loss train:', l.data.cpu().numpy())

        with torch.no_grad():
            y_pred_test = regressor(test_features)
            if (i + 1) % 50 == 0:
                print('epoch ', i, ', loss val:', loss(y_pred_test, test_target).data.cpu().numpy())

            val_smape, val_error = smape_metrics(test_target.detach().cpu().numpy(), y_pred_test.detach().cpu().numpy())
            if (i + 1) % 50 == 0 or (epochs - i < 10):
                print('epoch ', i, ', val metric:', val_smape, val_error)

            if val_smape < best_smape:
                best_smape = val_smape
                best_epoch = i
                save_linear(regressor, linear_file)

    print('best_smape, best_epoch:', best_smape, best_epoch)
    print("end")


def train_encoder_linear(fpath_train, fpath_test, if_nan,
                         training,
                         encode_pred_num, encode_window,
                         window_size, test_pred_num,
                         epochs, compute_representations,
                         storage_train, storage_test,
                         linear_save, model_save):

    train_generator, test_generator, train_nan, test_nan, train, test = get_data(fpath_train=fpath_train,
                                                                                 fpath_test=fpath_test,
                                                                                 if_nan=if_nan)
    encoder_linear_train(train_nan, train, test,
                         encode_pred_num=encode_pred_num, encode_window=encode_window,
                         window_size=window_size, test_pred_num=test_pred_num, epochs=epochs,
                         training=training, compute_representation=compute_representations,
                         storage_train=storage_train,
                         storage_test=storage_test,
                         linear_file=linear_save,
                         model_file=model_save)


if __name__ == '__main__':
    # train original encoder and linear for the dataset
    print('===========Main===========')
    # create a parser
    parser = argparse.ArgumentParser()

    # optimizer arguments
    parser.add_argument('--lr1', type=float, default=0.1)
    parser.add_argument('--train_epochs1', type=int, default=200)

    # training stage arguments
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--neg_samples', type=int, default=10)
    parser.add_argument('--compared_length', default=None)
    parser.add_argument('--compute_linear', default=True)
    args = parser.parse_args()

    fpath_nn5_train = 'NN5/daily.train'
    fpath_nn5_test = 'NN5/daily.test'
    path_m3_nn5_encoder = 'NN5/m3_nn5/m3_quarterly_train'
    path_m3_nn5_linear = 'NN5/m3_nn5/m3_quarterly_nn5_linear.pth'
    path_nn5_encoder = '../models_save/nn5_train'
    path_nn5_linear = '../models_save/nn5_train_linear.pth'

    test_pred_terms = 56

    #############################################
    #############################################
    compute_representations = True
    if_nan = False
    training = 0
    window_size = 1
    encode_pred_num = 56
    encode_window = 1
    epochs = 2000
    storage_train, storage_test = 'NN5/nn5_train_features', 'NN5/nn5_test_features'
    linear_save = path_nn5_linear
    model_save = path_nn5_encoder
    train_encoder_linear(fpath_nn5_train, fpath_nn5_test, if_nan,
                         training,
                         encode_pred_num, encode_window,
                         window_size, test_pred_terms,
                         epochs, compute_representations,
                         storage_train, storage_test,
                         linear_save, model_save)



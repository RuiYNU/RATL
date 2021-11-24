import torch
from encoder_models import Timeseries_Encoder as Model
import torch.nn as nn
import pickle as pkl
from utils import read_target
import numpy as np
from metrics import smape_metrics


def get_encoder_linear(path_encoder,
                       path_linear,
                       test_pred_term=56, mode=False):
    cuda, gpu = False, 0
    if torch.cuda.is_available():
        print("Using CUDA...")
        cuda = True

    parameters = {'compared_length': None, 'neg_samples': 10,
                  'batch_size': 1, 'train_steps': 300, 'lr': 0.001,
                  'n_cluster': 5, 'in_channels': 1, 'channels': 30,
                  'depth': 10, 'reduced_size': 80, 'kernel_size': 3,
                  'out_channels': 160, 'cuda': cuda, 'gpu': gpu, 'module': 'gru',
                  'enc_hidden': 1, 'ar_hidden': 24, 'prediction_step': 6}

    encoder = Model.CausalCNN_Time()
    encoder.set_params(**parameters)

    if mode:
        param = torch.load(path_encoder)
        encoder.encoder.load_state_dict(param['state_dict'])
    else:
        encoder.load_encoder(path_encoder)
    encoder.encoder.eval()

    if path_linear is None:
        linear = nn.Linear(160, test_pred_term)
        linear.eval()
    else:
        linear = nn.Linear(160, test_pred_term)
        linear.load_state_dict(torch.load(path_linear))
        linear.eval()
    return encoder, linear


def read_models(fpath_test='../dataset/M3/quarterly.test',
                storage_test='../dataset_save/m3c_test_quarterly_day.npy',
                path_encoder='../models_save/m3c_quarterly_train',
                path_linear='../models_save/m3c_quarterly_train_linear.pth',
                encode_pred_term=8, encode_window=1,
                test_pred_term=8,
                window_size=1,
                compute=True, mode=False):
    encoder, linear = get_encoder_linear(path_encoder, path_linear, test_pred_term, mode=mode)
    test = np.load(fpath_test + '.npy', allow_pickle=True)

    if compute:
        test_features = encoder.encode_window_new(test, encode_window, encode_pred_term, mode='test')

    else:
        with open(storage_test, 'rb') as f:
            test_features = pkl.load(f)


    test_features_day = torch.from_numpy(np.array(test_features))

    y_pred_test = linear(test_features_day)
    test_target = read_target(test, window_size, test_pred_term, mode='test')
    test_target = torch.from_numpy(test_target)

    val_smape, val_error = smape_metrics(test_target.detach().cpu().numpy(), y_pred_test.detach().cpu().numpy())
    print(val_smape)
    return val_smape


if __name__ == '__main__':
    ##################################################
    # read NN5 model (without transfer) smape: 0.2111
    fpath_nn5 = '../dataset/NN5/daily.test'
    storage_nn5_test = '../dataset_save/nn5_test.npy'
    path_nn5_encoder = '../models_save/nn5_train'
    path_nn5_linear = '../models_save/nn5_train_linear.pth'
    test_pred_term = 56
    window_size = 1
    encode_pred_term, encode_window = 1, 56
    read_models(fpath_test=fpath_nn5, storage_test=storage_nn5_test,
                path_encoder=path_nn5_encoder, path_linear=path_nn5_linear,
                encode_pred_term=encode_pred_term, encode_window=encode_window,
                test_pred_term=test_pred_term, window_size=window_size, compute=True)

    ########################################################
    # Use the model trained on M3_Quarterly for NN5 smape: 0.2254
    fpath_nn5 = '../dataset/NN5/daily.test'
    storage_m3_nn5_test = '../dataset/NN5/m3_nn5/m3_quarterly_encode_test'
    path_m3_nn5_encoder = '../dataset/NN5/m3_nn5/m3_quarterly_train'
    path_m3_nn5_linear = '../dataset/NN5/m3_nn5/m3_quarterly_nn5_linear.pth'

    encode_pred_term, encode_window = 1, 56
    test_pred_term = 56
    window_size = 1
    read_models(fpath_test=fpath_nn5, storage_test=storage_m3_nn5_test,
                path_encoder=path_m3_nn5_encoder, path_linear=path_m3_nn5_linear,
                encode_pred_term=encode_pred_term, encode_window=encode_window,
                test_pred_term=test_pred_term, window_size=window_size, compute=True)

    ######################################################################
    # Final model for NN5 (after transfer) smape: 0.2033
    model_save = '../models_save/nn5_encoder'
    linear_save = '../models_save/nn5_linear.pth'
    read_models(fpath_test=fpath_nn5, storage_test=storage_m3_nn5_test,
                path_encoder=model_save, path_linear=linear_save,
                encode_pred_term=encode_pred_term, encode_window=encode_window,
                test_pred_term=test_pred_term, window_size=window_size, compute=True, mode=True)






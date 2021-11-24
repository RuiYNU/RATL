import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import read_target
from metrics import smape_metrics
from transfer.get_encoder_linear import get_encoder_linear, read_models
import pickle as pkl
from utils import read_target
from metrics import smape_metrics
from dataset.train import save_linear
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import argparse
from dataset.train import get_data
from tqdm import tqdm


def construct_representation(batch, train, neg_samples=1, compared_length=None):
    """
    construct representations for positive and negative samples
    :param batch: batch
    :param train: training dataset
    :param neg_samples: negative samples
    :param compared_length:
    :return:
    """
    batch_size, train_size = batch.size(0), train.size(0)
    max_length = train.size(2)
    if compared_length is None:
        compared_length = np.inf

    # randomly select negative samples
    samples = torch.LongTensor(np.random.choice(train_size, size=(neg_samples, batch_size)))

    with torch.no_grad():
        lengths_batch = max_length - torch.sum(torch.isnan(batch[:, 0]), 1).data.cpu().numpy()

        lengths_neg_samples = np.empty((neg_samples, batch_size), dtype=int)
        # filter NAN get the original lengths of samples
        for i in range(neg_samples):
            lengths_neg_samples[i] = max_length - torch.sum(
                torch.isnan(train[samples[i], 0]), 1).data.cpu().numpy()

    # lengths of positive and negative samples
    pos_lengths = np.empty(batch_size, dtype=int)
    neg_lengths = np.empty((neg_samples, batch_size), dtype=int)

    for i in range(batch_size):
        pos_lengths[i] = np.random.randint(1, high=min(compared_length, lengths_batch[i])+1)
        for j in range(neg_samples):
            neg_lengths[j, i] = np.random.randint(1, high=min(compared_length, lengths_neg_samples[j, i])+1)

    selected_lengths = np.array([np.random.randint(pos_lengths[j],
                                                   high=min(compared_length, lengths_batch[j]) + 1)
                                 for j in range(batch_size)])

    # start index of selected samples (as anchors)
    selected_start_pos = np.array([np.random.randint(0,
                                                     high=lengths_batch[j]-selected_lengths[j] + 1)
                                   for j in range(batch_size)])

    # interval between positive and selected samples
    pos_interval = np.array([np.random.randint(0, high=selected_lengths[j]-pos_lengths[j]+1)
                             for j in range(batch_size)])
    pos_start_pos = selected_start_pos + pos_interval
    pos_end_pos = pos_start_pos + pos_lengths

    neg_start_pos = np.array([[np.random.randint(0,
                                                 high=lengths_neg_samples[i, j] - neg_lengths[i, j] + 1)
                               for j in range(batch_size)] for i in range(neg_samples)])

    selected_batch, pos_batch, neg_batch = [], [], []
    for j in range(batch_size):
        a = batch[j:j+1, :, selected_start_pos[j]:selected_start_pos[j]+selected_lengths[j]]
        selected_batch.append(a.clone().detach())

    for j in range(batch_size):
        a = batch[j:j+1, :, pos_start_pos[j]:pos_end_pos[j]]
        pos_batch.append(a.clone().detach())

    for i in range(neg_samples):
        for j in range(batch_size):
            a = train[samples[i, j]: samples[i, j] + 1][:, :, neg_start_pos[i, j]:
                                                              neg_start_pos[i, j] + neg_lengths[i, j]]
            neg_batch.append(a.clone().detach())
    return selected_batch, pos_batch, neg_batch


def test_val(fpath_train, fpath_test,
             encoder, linear,
             encode_pred_num, encode_window,
             test_pred_terms, window,
             compute_linear=True):

    train = np.load(fpath_train+'.npy', allow_pickle=True)
    test = np.load(fpath_test+'.npy', allow_pickle=True)

    encoder.encoder.eval()
    test_features = encoder.encode_window_new(test, encode_window, encode_pred_num, mode='test')
    test_features = torch.from_numpy(np.array(test_features))

    train_target = read_target(train, window, test_pred_terms, mode='train')
    test_target = read_target(test, window, test_pred_terms, mode='test')
    train_target = torch.from_numpy(train_target)
    test_target = torch.from_numpy(test_target)

    if compute_linear:
        regressor = linear
        y_pred_test = regressor(test_features)
        val_smape, _ = smape_metrics(test_target.detach().cpu().numpy(), y_pred_test.detach().cpu().numpy())
        print('ORI:', val_smape)

        # train training samples
        train_features = encoder.encode_window_new(train, encode_window, encode_pred_num, mode='train')
        train_features = torch.from_numpy(np.array(train_features))

        gpu = 0
        best_linear = regressor
        epochs = 3500
        regressor.double()
        if torch.cuda.is_available():
            regressor.cuda(gpu)
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)

        best_smape = 99999.0
        for i in range(epochs):
            y_pred_train = regressor(train_features)
            l = loss(y_pred_train, train_target)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                y_pred_test = regressor(test_features)

                val_smape, _ = smape_metrics(test_target.detach().cpu().numpy(), y_pred_test.detach().cpu().numpy())
                if (i+1) % 50 == 0:
                    print('VAL_EPOCH:{}, VAL_Metric:{}'.format(i, val_smape))

                if val_smape < best_smape:
                    best_smape = val_smape
                    best_linear = regressor

        val_smape, linear = best_smape, best_linear
    else:
        linear = linear.eval()
        y_pred_test = linear(test_features)
        val_smape, _ = smape_metrics(test_target.detach().cpu().numpy(),
                                     y_pred_test.detach().cpu().numpy())
    print('Best_smape:', val_smape)
    return val_smape, linear


def train_stage_1(args, fpath_train, fpath_test,
                  train, train_data_generator,
                  source_encoder, target_encoder, target_linear,
                  model_save_path, linear_save):

    KL_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(target_encoder.encoder.parameters(), lr=0.001)
    # build a scheduler
    scheduler1 = CosineAnnealingLR(optimizer, args.train_epochs1, 0.1 * args.lr1)

    best_val_acc = 9999
    train = torch.from_numpy(train)

    for epoch in range(1, args.train_epochs1):

        training_loss, triplets_num = 0, 0

        total = train_data_generator.__len__()
        bar = tqdm(total=total, desc='stage1: epoch %d' % (epoch), unit='batch')

        for index, batch in enumerate(train_data_generator):
            bar.update(1)

            batch_selected, batch_pos, batch_neg = construct_representation(batch, train,
                                                                            args.neg_samples,
                                                                            args.compared_length)

            with torch.no_grad():
                source_selected = torch.cat([source_encoder.encoder(batch_selected[j])
                                             for j in range(batch.size(0))])
                source_pos = torch.cat([source_encoder.encoder(batch_pos[j])
                                        for j in range(batch.size(0))])
                source_neg = torch.cat([source_encoder.encoder(batch_neg[j])
                                        for j in range(batch.size(0))])

                source_selected = F.normalize(source_selected, p=2, dim=1)
                source_pos = F.normalize(source_pos, p=2, dim=1)
                source_neg = F.normalize(source_neg, p=2, dim=1)

            target_selected = torch.cat([target_encoder.encoder(batch_selected[j])
                                         for j in range(batch.size(0))])
            target_pos = torch.cat([target_encoder.encoder(batch_pos[j])
                                    for j in range(batch.size(0))])
            target_neg = torch.cat([target_encoder.encoder(batch_neg[j])
                                    for j in range(batch.size(0))])
            target_selected = F.normalize(target_selected, p=2, dim=1)
            target_pos = F.normalize(target_pos, p=2, dim=1)
            target_neg = F.normalize(target_neg, p=2, dim=1)

            # compute distance between selected and positive samples
            source_sel_pos_dist = torch.norm(source_selected - source_pos, p=2, dim=1)
            source_sel_neg_dist = torch.norm(source_selected - source_neg, p=2, dim=1)

            target_sel_pos_dist = torch.norm(target_selected - target_pos, p=2, dim=1)
            target_sel_neg_dist = torch.norm(target_selected - target_neg, p=2, dim=1)

            source_distribution = torch.sigmoid((source_sel_neg_dist-source_sel_pos_dist))
            target_distribution = torch.sigmoid((target_sel_neg_dist - target_sel_pos_dist))
            source_distribution_1 = torch.cat([source_distribution.unsqueeze(1), 1-source_distribution.unsqueeze(1)])
            target_distribution_1 = torch.cat([target_distribution.unsqueeze(1), 1-target_distribution.unsqueeze(1)])

            loss_value = 1000 * KL_loss(torch.log(target_distribution_1), source_distribution_1)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            training_loss += loss_value.cpu().item() * target_distribution.size()[0]
            triplets_num += target_distribution.size()[0]

        training_loss /= triplets_num
        print('batch:{}  loss:{} n_triplets:{}'.format(index, training_loss, triplets_num))
        bar.close()

        if epoch % 5 == 0:
            val_acc, target_linear_new = test_val(fpath_train, fpath_test,
                                                  target_encoder, target_linear,
                                                  encode_pred_num=args.encode_pred_num, encode_window=args.encode_window,
                                                  test_pred_terms=args.test_pred_term, window=args.window,
                                                  compute_linear=args.compute_linear)
            target_linear = target_linear_new
            print('epoch:{}, val_acc:{}'.format(epoch, val_acc))
            if val_acc < best_val_acc:
                best_val_acc = val_acc
                record = {'state_dict': target_encoder.encoder.state_dict(),
                          'val_smape:': val_acc}
                torch.save(record, model_save_path)
                save_linear(target_linear, linear_save)
                print('save_model')
        else:
            scheduler1.step()
    return target_linear


def train_stage_2(args, best_smape, fpath_train, fpath_test,
                  path_target_encoder, path_target_linear,
                  path_source_encoder, path_source_linear,
                  storage_train, storage_test, linear_save):

    train = np.load(fpath_train + '.npy', allow_pickle=True)
    test = np.load(fpath_test + '.npy', allow_pickle=True)

    target_encoder, target_linear = get_encoder_linear(path_encoder=path_target_encoder,
                                                       path_linear=path_target_linear,
                                                       test_pred_term=args.test_pred_term,
                                                       mode=args.mode_2)
    _, source_linear = get_encoder_linear(path_source_encoder, path_source_linear,
                                          test_pred_term=args.test_pred_term, mode=args.mode_1)
    target_encoder.encoder.eval()
    source_linear.eval()

    if args.compute_2:
        train_features = target_encoder.encode_window_new(train, args.encode_window,
                                                          args.encode_pred_num, 'train')
        test_features = target_encoder.encode_window_new(test, args.encode_window,
                                                         args.encode_pred_num, 'test')
        with open(storage_train, 'wb') as f:
            pkl.dump([train_features], f)
        with open(storage_test, 'wb') as f:
            pkl.dump([test_features], f)
    else:
        with open(storage_train, 'rb') as f:
            train_features = pkl.load(f)

        with open(storage_test, 'rb') as f:
            test_features = pkl.load(f)

    train_features = torch.from_numpy(np.array(train_features))
    test_features = torch.from_numpy(np.array(test_features))

    train_target = read_target(train, args.window, args.test_pred_term, mode='train')
    test_target = read_target(test, args.window, args.test_pred_term, mode='test')
    train_target = torch.from_numpy(train_target)
    test_target = torch.from_numpy(test_target)

    target_linear.double()
    if torch.cuda.is_available():
        target_linear.cuda(0)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(target_linear.parameters(), lr=0.001)

    #best_smape = 99999.0
    for i in range(args.train_epochs2):
        y_pred_train = target_linear(train_features)
        with torch.no_grad():
            y_pred_train_source = source_linear(train_features)
            y_source_exp = torch.exp(-(y_pred_train_source - train_target) ** 2)

        y_target_exp = torch.exp(-(y_pred_train - train_target) ** 2)
        w = (y_source_exp / (y_source_exp + y_target_exp)).mean(1)
        w = F.normalize(w, p=2, dim=0)
        l2 = w * (((y_pred_train - y_pred_train_source) ** 2).mean(1))
        l = loss(y_pred_train, train_target)

        l_tol = l+l2.mean()
        l_tol.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 50 == 0:
            print('epoch:{}, loss train:{}'.format(i, l.data.cpu().numpy()))

        with torch.no_grad():
            y_pred_test = target_linear(test_features)
            if(i+1) % 50 == 0:
                print('epoch:{}, loss val:{}'.format(i, loss(y_pred_test, test_target).data.cpu().numpy()))
            val_smape, val_err = smape_metrics(test_target.detach().cpu().numpy(),
                                               y_pred_test.detach().cpu().numpy())
            if (i+1) % 50 == 0:
                print('epoch:{}, SMAPE:{}'.format(i, val_smape))

            if val_smape < best_smape:
                best_smape = val_smape
                save_linear(target_linear, linear_save)


def train_transfer(args, fpath_train, fpath_test,
                   path_source_encoder, path_source_linear,
                   path_target_encoder, path_target_linear,
                   storage_train_2, storage_test_2,
                   model_save_path, linear_save,
                   if_nan=False):

    source_encoder, source_linear = get_encoder_linear(path_source_encoder, path_source_linear,
                                                       test_pred_term=args.test_pred_term,
                                                       mode=args.mode_1)
    target_encoder, target_linear = get_encoder_linear(path_target_encoder, path_target_linear,
                                                       args.test_pred_term,
                                                       mode=args.mode_1)


    train_generator, test_generator, train_nan, test_nan, train, test = get_data(fpath_train=fpath_train,
                                                                                 fpath_test=fpath_test,
                                                                                 if_nan=if_nan)

    print('#############################Stage_1############################')
    train_stage_1(args, fpath_train, fpath_test,
                  train_nan, train_generator,
                  source_encoder, target_encoder, target_linear,
                  model_save_path, linear_save+'_1')

    best_smape = read_models(fpath_test=fpath_test, storage_test=storage_test_2,
                path_encoder=model_save_path, path_linear=linear_save+'_1_linear.pth',
                encode_pred_term=args.encode_pred_num, encode_window=args.encode_window,
                test_pred_term=args.test_pred_term, window_size=args.window,
                compute=True, mode=args.mode_2)

    print('#############################Stage_2############################')

    train_stage_2(args, best_smape, fpath_train, fpath_test,
                  model_save_path, linear_save+'_1_linear.pth',
                  path_source_encoder, path_source_linear,
                  storage_train_2, storage_test_2, linear_save)


if __name__ == '__main__':
    print('===========Main===========')
    # create a parser
    parser = argparse.ArgumentParser()

    # optimizer arguments
    parser.add_argument('--lr1', type=float, default=0.1)
    parser.add_argument('--train_epochs1', type=int, default=200)

    # training stage arguments
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--neg_samples', type=int, default=10)
    parser.add_argument('--compared_length', default=50)
    parser.add_argument('--compute_linear', default=True)
    parser.add_argument('--mode_1', default=False)

    parser.add_argument('--train_epochs2', type=int, default=1000)
    parser.add_argument('--compute_2', default=True)
    parser.add_argument('--mode_2', default=True)

    parser.add_argument('--encode_pred_num', type=int, default=1)
    parser.add_argument('--encode_window', type=int, default=56)
    parser.add_argument('--test_pred_term', type=int, default=56)

    args = parser.parse_args()

    fpath_nn5_train = '../dataset/NN5/daily.train'
    fpath_nn5_test = '../dataset/NN5/daily.test'
    path_m3_nn5_encoder = '../dataset/NN5/m3_nn5/m3c_quarterly_train'
    path_m3_nn5_linear = '../dataset/NN5/m3_nn5/m3_quarterly_nn5_linear.pth'

    path_nn5_encoder = '../models_save/nn5_train'
    path_nn5_linear = '../models_save/nn5_train_linear.pth'

    model_save_path = '../models_save/nn5_encoder'
    linear_save = '../models_save/nn5'
    storage_train_2 = '../dataset/NN5/m3_nn5/m3c_quarterly_train_stage_2'
    storage_test_2 = '../dataset/NN5/m3_nn5/m3c_quarterly_test_stage_2'

    train_transfer(args, fpath_nn5_train, fpath_nn5_test,
                   path_m3_nn5_encoder, path_m3_nn5_linear,
                   path_nn5_encoder, path_nn5_linear,
                   storage_train_2, storage_test_2,
                   model_save_path, linear_save,
                   if_nan=False)













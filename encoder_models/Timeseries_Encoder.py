import torch.utils.data
import sklearn
import sklearn.svm
import datetime
import sys
import losses as losses
import math
import networks as networks
import numpy as np
from utils import Dataset, LabelledDataset
from losses import clustering_loss, diff_loss, Sequential_loss, CPC_loss2


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


class TimeSeriesEncoder(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    '''
    @param compared_length: length of sub_series used in losses, if None, compared_length=length(X)
    @param nb_random_samples: number of negative samples used in losses
    @param batch_size
    @param train_steps: optimization steps in training
    '''
    def __init__(self, compared_length, neg_samples,
                 batch_size, train_steps, encoder, in_channels, out_channels, lr, n_cluster,
                 paramers,
                 cuda=False, gpu=0, module='gru',
                 enc_hidden=1, ar_hidden=32, prediction_step=3):
        self.compared_length = compared_length
        self.neg_samples = neg_samples
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.encoder = encoder
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr = lr
        self.n_cluster = n_cluster
        self.parameters = paramers
        self.cuda = cuda
        self.gpu = gpu

        self.module = module
        self.enc_hidden = enc_hidden
        self.ar_hidden = ar_hidden
        self.prediction_step = prediction_step


        self.cpc_loss_2 = CPC_loss2.CPC(module=self.module, encoder=self.encoder,
                                     enc_hidden=self.enc_hidden, ar_hidden=self.ar_hidden,
                                     pred_num=self.prediction_step, batch_size=self.batch_size,
                                     neg_samples=self.neg_samples)

        self.seq_loss = Sequential_loss.Seq_loss(encoder=self.encoder, enc_hidden=self.out_channels,
                                                 compared_length=self.compared_length)
        self.diff_loss = diff_loss.diff_loss(encoder=self.encoder, compared_length=self.compared_length,
                                             neg_samples=self.neg_samples)


        n_warmup_steps = 1000
        self.optimizer = ScheduledOptim(torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.encoder.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
            n_warmup_steps)

    def encode(self, x, batch_size):
        x_dataset = Dataset(x)
        x_generator = torch.utils.data.DataLoader(x_dataset, batch_size)
        features = np.zeros((np.shape(x)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        is_varying = bool(np.isnan(np.sum(x)))

        with torch.no_grad():
            i = 0
            if not is_varying:
                for batch in x_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[i * batch_size: (i + 1) * batch_size] = self.encoder(batch).cpu()
                    i += 1
            else:
                for batch in x_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)

                    lengths_batch = batch.size(2) - torch.sum(torch.isnan(batch[:, 0]), 1).data.cpu().numpy()

                    for j in range(batch.size(0)):
                        batch_j = batch[j, :, :lengths_batch[j]]
                        #batch_j = batch_j.reshape((1, batch_j.shape[0], -1))
                        #print('batch_J:', batch_j.shape)
                        features[j:j+1] = self.encoder(batch_j).cpu()

        self.encoder = self.encoder.train()
        return features

    def fit_encoder(self, x):
        is_varying = bool(np.isnan(np.sum(x)))
        train = torch.from_numpy(x)
        if self.cuda:
            train = train.cuda(self.gpu) #(1,1,100)

        x_dataset = Dataset(x)
        x_generator = torch.utils.data.DataLoader(x_dataset, batch_size=self.batch_size, shuffle=True)  #drop_last=True  by yq


        centroids_init = clustering_loss.init_centroids(self.encoder, train, self.n_cluster, is_varying)
        self.cluster_loss = clustering_loss.TemporalClustering(self.encoder, self.n_cluster, centroids_init)


        i, epoch = 0, 0
        #self.cpc_loss_2.train()

        while i < self.train_steps:
            for batch in x_generator:
                #hidden = self.cpc_loss_2.init_hidden(len(batch), use_gpu=False)
                hidden = self.cpc_loss_2.init_hidden(1, use_gpu=False)
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not is_varying:
                    loss_cpc = self.cpc_loss_2(batch, train, hidden)
                    loss_diff = self.diff_loss(train, batch, self.encoder, varying=is_varying)
                    loss_seq = self.seq_loss(train, batch, self.encoder, varying=is_varying)
                    loss_cluster = self.cluster_loss(batch, varying=is_varying)
                    #loss_total = loss_diff + loss_cpc + loss_seq + loss_cluster
                    loss_total = loss_diff*0.1 + loss_cpc + loss_seq*0.1 + loss_cluster
                else:
                    loss_cpc = self.cpc_loss_2(batch, train, hidden, varying=is_varying)
                    loss_diff = self.diff_loss(train, batch, self.encoder, varying=is_varying)
                    loss_seq = self.seq_loss(train, batch, self.encoder, varying=is_varying)
                    loss_cluster = self.cluster_loss(batch, varying=is_varying)
                    #loss_total = loss_diff + loss_cpc + loss_seq + loss_cluster
                    loss_total = loss_diff*0.1 + loss_cpc + loss_seq*0.1 + loss_cluster

                loss_total.backward()
                self.optimizer.step()
                i += 1

                if i%5==0:
                    print('{}:step:{}/{}, loss:{}'.format(datetime.datetime.now().isoformat(), i + 1, self.train_steps,
                                                           loss_total.detach().cpu().numpy()))

                    sys.stdout.flush()
                if i >= self.train_steps:
                    break
          # (1,1,100)
        return self.encoder

    def save_encoder(self, file_name):
        torch.save(
            self.encoder.state_dict(),
            file_name + '_encoder.pth'
        )

    def load_encoder(self, file_name):
        self.encoder.load_state_dict(torch.load(
            file_name + '_encoder.pth',
            map_location=lambda storage, loc: storage
        ))

    def encode_window_new(self, X, window, test_pred_num, mode):
        #print('X_shape:', X.shape)
        features_list = []
        masking = np.empty((
            1,
            1, window
        ))  # min(window_batch_size, 200) #numpy.shape(X)[2] - window + 1; numpy.shape(X)[1]
        for b in range(np.shape(X)[0]):
            if b % 20 == 0:
                print('encoding num:{}/{}'.format(b, np.shape(X)[0]))

            if mode == 'train':
                for i in range(0, X[b].shape[0]-window - test_pred_num):
                    masking[0, 0, :] = X[b][i:i+window] #(200, 1, 10)
                    aaa = np.swapaxes(
                        self.encode(masking, batch_size=1), 0, 1
                    )  #(160, 1)

                    features_list.append(aaa.squeeze()) # squeeze-->160

            elif mode == 'test':
                i = X[b].shape[0] - test_pred_num - window

                masking[0, 0, :] = X[b][i:i + window]  # (200, 1, 10)
                aaa = np.swapaxes(
                    self.encode(masking, batch_size=1), 0, 1
                )  # (160, 1)

                features_list.append(aaa.squeeze())

        return np.array(features_list)


class CausalCNN_Time(TimeSeriesEncoder):

    def __init__(self, compared_length=20, neg_samples=10, batch_size=1, train_steps=200,
                 lr=0.001, n_cluster=5, in_channels=1, channels=10, depth=1, reduced_size=10, out_channels=64,
                 kernel_size=4, cuda=False, gpu=0, module='gru', enc_hidden=1, ar_hidden=2, prediction_step=6):

        super(CausalCNN_Time, self).__init__(compared_length, neg_samples, batch_size, train_steps,
                                        self.__create_encoder(in_channels, channels, depth, reduced_size,
                                                              out_channels, kernel_size, cuda, gpu),
                                        in_channels, out_channels, lr, n_cluster,
                                        self.__get_encoder_parameters(in_channels, channels, depth,
                                                                      reduced_size, out_channels, kernel_size
                                                                      ),
                                        cuda, gpu, module, enc_hidden, ar_hidden, prediction_step)

        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size
        self.enc_hidden = enc_hidden
        self.ar_hidden = ar_hidden
        self.prediction_step = prediction_step

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = networks.causal_cnn.CausalCNNEncoder(in_channels, channels, depth,
                                            reduced_size, out_channels, kernel_size)
        if cuda:
            encoder.cuda(gpu)

        encoder.double()
        return encoder

    def __get_encoder_parameters(self, in_channels, channels, depth,
                                 reduced_size, out_channels, kernel_size):
        return {'in_channels': in_channels, 'channels': channels, 'depth': depth,
                'reduced_size': reduced_size, 'out_channels': out_channels,
                'kernel_size': kernel_size}

    def set_params(self, compared_length, neg_samples,
                   batch_size, train_steps, lr, n_cluster, in_channels,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   cuda, gpu, module='gru',
                   enc_hidden=1, ar_hidden=2, prediction_step=6):

        self.__init__(
            compared_length, neg_samples, batch_size,
            train_steps, lr, n_cluster, in_channels, channels, depth,
            reduced_size, out_channels, kernel_size, cuda, gpu, module,
            enc_hidden, ar_hidden, prediction_step)
        return self

    def get_params(self, deep=True):
        return {
            'compared_length': self.cpc_loss_2.compared_length,
            'neg_samples': self.cpc_loss_2.neg_samples,
            'batch_size': self.batch_size,
            'train_steps': self.train_steps,
            'lr': self.lr,
            'n_cluster': self.n_cluster,
            'in_channels': self.in_channels,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu,
            'module': self.module,
            'enc_hidden': self.enc_hidden,
            'ar_hidden': self.ar_hidden,
            'prediction_step': self.prediction_step
        }






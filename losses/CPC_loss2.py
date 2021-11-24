import torch
import torch.nn as nn
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)


class CPC(torch.nn.Module):
    def __init__(self, module, encoder, enc_hidden, ar_hidden,
                 pred_num, batch_size, neg_samples=10):
        super(CPC, self).__init__()
        self.module = module
        self.encoder = encoder
        self.enc_hidden = enc_hidden
        self.ar_hidden = ar_hidden
        self.pred_num = pred_num
        self.batch_size = batch_size
        if self.module == 'lstm':
            self.gru = nn.LSTM(self.enc_hidden, self.ar_hidden, batch_first=True)
        else:
            self.gru = nn.GRU(self.enc_hidden, self.ar_hidden, num_layers=1, bidirectional=False, batch_first=True)

        self.Wk = nn.ModuleList([nn.Linear(self.ar_hidden, self.enc_hidden) for i in range(self.pred_num)])
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.mseloss = nn.MSELoss(size_average=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden = torch.zeros(1, self.batch_size, self.ar_hidden, dtype=torch.double).to(self.device)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, self.ar_hidden).cuda()
        else:
            return torch.zeros(1, batch_size, self.ar_hidden)

    def forward(self, batch, train, hidden, varying=False):
        batch_size = batch.size(0)
        train_size = train.size(0)  # train_nums
        max_length = train.size(2)

        if varying:
            with torch.no_grad():
                lengths_batch = max_length - torch.sum(torch.isnan(batch[:, 0]), 1).data.cpu().numpy()
                lengths_selected = np.empty(batch_size, dtype=int)

                for j in range(batch_size):
                    lengths_selected[j] = np.random.randint(
                        self.pred_num*2, high=lengths_batch[j]+1)

            start_ind_selected = np.array([np.random.randint(0, high=lengths_batch[j] - lengths_selected[j] + 1
                                                             ) for j in range(batch_size)])

            z = torch.cat([self.encoder(batch[j: j + 1, :,
                                        start_ind_selected[j]: start_ind_selected[j] + lengths_selected[j]]
                                        ) for j in range(batch_size)])

            if len(z.shape) < 3:
                z = z[:, :, np.newaxis]

            t_pos = np.random.randint(0, high=z.shape[1] - self.pred_num, size=batch_size)


            loss_nce = 0
            encode_samples = torch.empty((self.pred_num, batch_size, z.shape[-1])).double()
            for i in range(1, self.pred_num + 1):
                for j in range(batch_size):
                    encode_samples[i - 1, j, :] = z[j, t_pos[j] + i, :].view(1, z.shape[-1])

            forward_seq = []
            for j in range(batch_size):
                forward_seq.append(z[j, :t_pos[j] + 1, :])

            c_t = torch.empty((batch_size, 1, self.ar_hidden))

            for j in range(batch_size):
                forward_seq_j = forward_seq[j][np.newaxis, :, :]
                output_j, hidden_j = self.gru(forward_seq_j, hidden)
                c_t[j] = output_j[:, t_pos[j], :]
            c_t = c_t.view(batch_size, self.ar_hidden)
            pred = torch.empty((self.pred_num, batch_size, self.enc_hidden)).double()

            for i in range(self.pred_num):
                linear = self.Wk[i]
                pred[i] = linear(c_t)

            for i in range(self.pred_num):
                # print('encode_samples:', encode_samples[i].shape, pred[i].shape)
                total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))

                xx1 = self.logsoftmax(total)
                xx2 = torch.diag(xx1)
                xx3 = torch.sum(xx2)
                loss_nce += xx3

            loss_nce /= -1. * batch_size * self.pred_num
            return loss_nce
        else:
            lengths_selected = np.random.randint(self.pred_num*2, high=max_length + 1)
            start_ind_selected = np.random.randint(
                0, high=max_length - lengths_selected + 1, size=batch_size)


            z = torch.cat([self.encoder(batch[j: j + 1, :,
                                        start_ind_selected[j]: start_ind_selected[j] + lengths_selected]
                                        ) for j in range(batch_size)]) # batch_size * enc_hidden
            if len(z.shape) < 3:
                z = z[:, :, np.newaxis]

            t_pos = np.random.randint(0, high=z.shape[1] - self.pred_num, size=batch_size)

            loss_nce = 0
            encode_samples = torch.empty((self.pred_num, batch_size, z.shape[-1])).double()

            for i in range(1, self.pred_num + 1):
                for j in range(batch_size):
                    encode_samples[i - 1, j, :] = z[j, t_pos[j] + i, :].view(1, z.shape[-1])

            forward_seq = []
            for j in range(batch_size):
                forward_seq.append(z[j, :t_pos[j] + 1, :])
                #print(forward_seq[j].shape)

            c_t = torch.empty((batch_size, 1, self.ar_hidden))
            for j in range(batch_size):
                forward_seq_j = forward_seq[j][np.newaxis, :, :]
                output_j, hidden_j = self.gru(forward_seq_j, hidden)
                c_t[j] = output_j[:, t_pos[j], :]
            c_t = c_t.view(batch_size, self.ar_hidden)
            pred = torch.empty((self.pred_num, batch_size, self.enc_hidden)).double()

            for i in range(self.pred_num):
                linear = self.Wk[i]
                pred[i] = linear(c_t)

            for i in range(self.pred_num):
                # print('encode_samples:', encode_samples[i].shape, pred[i].shape)
                total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))

                xx1 = self.logsoftmax(total)
                xx2 = torch.diag(xx1)
                xx3 = torch.sum(xx2)
                loss_nce += xx3

            loss_nce /= -1. * batch_size * self.pred_num
            return loss_nce



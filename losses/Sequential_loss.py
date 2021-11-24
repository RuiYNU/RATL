import torch
import torch.nn as nn
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)

class Seq_loss(torch.nn.Module):
    def __init__(self, encoder, enc_hidden, compared_length=None):
        super(Seq_loss, self).__init__()
        self.encoder = encoder
        #self.neg_samples = neg_samples
        self.compared_length = compared_length
        self.enc_hidden = enc_hidden
        if compared_length is None:
            self.compared_length = np.inf

        self.loss = nn.BCEWithLogitsLoss()
        self.predictor = nn.Linear(self.enc_hidden, 2, bias=False)

    def forward(self, train, batch, encoder, varying=True):
        batch_size = batch.size(0)
        length = min(self.compared_length, train.size(2))

        if varying:
            with torch.no_grad():
                lengths_batch = length - torch.sum(torch.isnan(batch[:, 0]), 1).data.cpu().numpy()

            lengths_pos = np.empty(batch_size, dtype=int)
            for j in range(batch_size):
                lengths_pos[j] = np.random.randint(
                    1, high=min(self.compared_length, lengths_batch[j]) + 1)

            # start points of sub_series from different series
            start_ind_pos = np.array([np.random.randint(0, high=lengths_batch[j] - lengths_pos[j] + 1
                                                             ) for j in range(batch_size)])
            for j in range(batch_size):
                if j == 0:
                    x_pos = batch[j:j+1, :, start_ind_pos[j]:start_ind_pos[j]+lengths_pos[j]]
                    x_neg = x_pos[:, torch.randperm(x_pos.size(1))]
                    Z_pos = encoder(x_pos)
                    Z_neg = encoder(x_neg)
                else:
                    x_pos_new = batch[j:j + 1, :, start_ind_pos[j]:start_ind_pos[j] + lengths_pos[j]]
                    x_neg_new = x_pos_new[:, torch.randperm(x_pos_new.size(1))]
                    Z_pos_new = encoder(x_pos_new) # 1 * enc_hidden
                    Z_neg_new = encoder(x_neg_new)
                    Z_pos, Z_neg = torch.cat((Z_pos, Z_pos_new)), torch.cat((Z_neg, Z_neg_new))

        else:
            # length of sub_series (can be varying)
            length_pos = np.random.randint(1, high=length + 1)
            start_ind_pos = np.random.randint(
                0, high=length - length_pos + 1, size=batch_size)

            for j in range(batch_size):
                if j == 0:
                    x_pos = batch[j:j + 1, :, start_ind_pos[j]: start_ind_pos[j] + length_pos]
                    x_neg = x_pos[:, torch.randperm(x_pos.size(1))]
                    Z_pos = self.encoder(x_pos)
                    Z_neg = self.encoder(x_neg)
                else:
                    x_pos_new = batch[j:j + 1, :, start_ind_pos[j]: start_ind_pos[j] + length_pos]
                    x_neg_new = x_pos_new[:, torch.randperm(x_pos_new.size(1))]
                    Z_pos_new = self.encoder(x_pos_new)
                    Z_neg_new = self.encoder(x_neg_new)
                    Z_pos, Z_neg = torch.cat((Z_pos, Z_pos_new)), torch.cat((Z_neg, Z_neg_new))


        L_pos_neg = torch.ones(Z_pos.size(0) + Z_neg.size(0))
        L_pos_neg[:Z_neg.size(0)] = 0
        Z_pos_neg = torch.cat((Z_neg, Z_pos)) # 2*batch_size * enc_hidden

        rand_ind = torch.randperm(Z_pos_neg.size(0))
        Z_pos_neg = Z_pos_neg[rand_ind, :]
        L_pos_neg = L_pos_neg[rand_ind]

        #print('L:', L_pos_neg.long().unsqueeze(1).shape)

        L = torch.zeros(L_pos_neg.shape[0], 2).scatter_(1, L_pos_neg.long().unsqueeze(1), 1)
        #print('L:', L_pos_neg)
        #print('L_m:', L)

        Z_output = self.predictor(Z_pos_neg)
        loss = self.loss(Z_output, L)
        return loss








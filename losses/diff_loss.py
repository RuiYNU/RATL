import torch
import torch.nn as nn
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)


class diff_loss(torch.nn.Module):
    def __init__(self, encoder, compared_length=None, neg_samples=1):
        super(diff_loss, self).__init__()
        self.compared_length = compared_length
        if compared_length is None:
            self.compared_length = np.inf

        self.encoder = encoder
        self.neg_samples = neg_samples
        self.loss = nn.LogSoftmax(dim=1)

    def forward(self, train, batch, encoder, varying=True):
        batch_size = batch.size(0)
        train_size = train.size(0) # train_nums
        max_length = train.size(2)
        length = train.size(2)

        samples = np.random.choice(
            train_size, size=(self.neg_samples, batch_size)
        )
        # randomly select neg_samples samples from train_data,
        # for negative samples generation (series from different series)
        samples = torch.LongTensor(samples)

        if varying:
            with torch.no_grad():
                lengths_batch = max_length - torch.sum(torch.isnan(batch[:, 0]), 1).data.cpu().numpy()
                lengths_samples = np.empty((self.neg_samples, batch_size), dtype=int)

                for i in range(self.neg_samples):
                    lengths_samples[i] = max_length - torch.sum(torch.isnan(train[samples[i], 0]), 1
                                                                ).data.cpu().numpy()

            lengths_pos = np.empty(batch_size, dtype=int)
            lengths_neg = np.empty(
                (self.neg_samples, batch_size), dtype=int
            )
            for j in range(batch_size):
                lengths_pos[j] = np.random.randint(
                    1, high=min(self.compared_length, lengths_batch[j]) + 1)
                for i in range(self.neg_samples):
                    lengths_neg[i, j] = np.random.randint(
                        1, high=min(self.compared_length, lengths_samples[i, j]) + 1)
            # lengths of selected sub_series(varying)
            length_selected = np.array([np.random.randint(lengths_pos[j],
                                                      high=min(self.compared_length, lengths_batch[j]) + 1
                                                      ) for j in range(batch_size)])

            # start points of selected sub_series
            start_ind_selected = np.array([np.random.randint(0, high=lengths_batch[j] - length_selected[j] + 1
                                                             ) for j in range(batch_size)])

            # interval
            start_ind_pos = np.array([np.random.randint(0, high=length_selected[j] - lengths_pos[j] + 1
                                                        ) for j in range(batch_size)])
            start_pos = start_ind_selected + start_ind_pos
            end_pos = start_pos + lengths_pos

            start_ind_neg = np.array([[np.random.randint(0, high=lengths_samples[i, j] - lengths_neg[i, j] + 1
            ) for j in range(batch_size)] for i in range(self.neg_samples)])

            Z = torch.cat([encoder(batch[j: j + 1, :,
                                   start_ind_selected[j]: start_ind_selected[j] + length_selected[j]]
                                   ) for j in range(batch_size)])  # selected representations

            Z_pos = torch.cat([encoder(batch[j: j + 1, :,
                                       end_pos[j] - lengths_pos[j]: end_pos[j]]
                                       ) for j in range(batch_size)])

            Z_size = Z.size(1)
            pos_loss = torch.squeeze(torch.bmm(Z.view(batch_size, 1, Z_size),
                                               Z_pos.view(batch_size, Z_size, 1)), 1)  # batch_size * 1
            neg_loss = torch.zeros_like(pos_loss)

            # sub_series from different series (negative samples of the selected sub_series)
            for i in range(self.neg_samples):
                Z_neg = torch.cat([encoder(train[samples[i, j]: samples[i, j] + 1][:, :,
                                           start_ind_neg[i, j]: start_ind_neg[i, j] + lengths_neg[i, j]]
                                           ) for j in range(batch_size)])
                neg_loss = torch.cat((neg_loss, torch.squeeze(torch.bmm(Z.view(batch_size, 1, Z_size),
                                                                        Z_neg.view(batch_size, Z_size, 1)), 1)),1)
                # batch_size * (1+neg_samples)
            neg_loss = neg_loss[:, 1:]
            pos_neg_loss = torch.cat((pos_loss, neg_loss), 1)
            loss = -self.loss(pos_neg_loss)[:, 0]
        else:
            # lengths of the selected sample and its positive and negative samples can be varying
            length_comp = np.random.randint(1, high=length + 1)  # length of positive and negative samples
            length_selected = np.random.randint(
                length_comp, high=length + 1
            )  # Length of selected sample
            start_ind_selected = np.random.randint(
                0, high=length - length_selected + 1, size=batch_size)
            start_ind_pos = np.random.randint(
                0, high=length_selected - length_comp + 1, size=batch_size
            )  # interval between pos and selected samples
            start_pos = start_ind_selected + start_ind_pos
            end_pos = start_pos + length_comp

            start_ind_neg = np.random.randint(
                0, high=length - length_comp + 1,
                size=(self.neg_samples, batch_size)
            )

            Z = encoder(torch.cat(
                [batch[
                 j: j + 1, :,
                 start_ind_selected[j]: start_ind_selected[j] + length_selected
                 ] for j in range(batch_size)]
            ))

            Z_pos = encoder(torch.cat(
                [batch[
                 j: j + 1, :, end_pos[j] - length_comp: end_pos[j]
                 ] for j in range(batch_size)]
            ))

            Z_size = Z.size(1)

            pos_loss = torch.squeeze(torch.bmm(Z.view(batch_size, 1, Z_size),
                                               Z_pos.view(batch_size, Z_size, 1)), 1)  # batch_size * 1
            neg_loss = torch.zeros_like(pos_loss)

            # construct neg_samples (sub_series from different series)
            for i in range(self.neg_samples):
                Z_neg = encoder(torch.cat([train[samples[i, j]: samples[i, j] + 1][
                                           :, :,
                                           start_ind_neg[i, j]:
                                           start_ind_neg[i, j] + length_comp
                                           ] for j in range(batch_size)]))
                neg_loss = torch.cat((neg_loss, torch.squeeze(torch.bmm(Z.view(batch_size, 1, Z_size),
                                                                        Z_neg.view(batch_size, Z_size, 1)), 1)),
                                     1)  # batch_size * (1+neg_samples)
            neg_loss = neg_loss[:, 1:]
            pos_neg_loss = torch.cat((pos_loss, neg_loss), 1)
            loss = -self.loss(pos_neg_loss)[:, 0]

        loss = loss.sum() / batch_size
        return loss








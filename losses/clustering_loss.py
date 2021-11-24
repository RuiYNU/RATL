import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F
torch.set_default_tensor_type(torch.DoubleTensor)


def compute_similarity(z, centroids):
    # z: batch_size * enc_hidden  centroids: n_centers * enc_hidden
    n_clusters, enc_hidden = centroids.shape[0], centroids.shape[1]
    z = z.expand((n_clusters, z.shape[0], enc_hidden))
    mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
    return torch.transpose(mse, 0, 1)


def init_centroids(encoder, x, n_clusters, is_varying):
    if is_varying:
        max_length = x.shape[-1]

        for i in range(x.shape[0]):
            # filter NAN
            lengths_i = max_length - torch.sum(
                torch.isnan(x[i, :]), 1
            ).data.cpu().numpy()
            len_i = int(lengths_i)
            xi = x[i, :, :len_i]
            if len(xi.shape)<3:
                xi = xi[:, np.newaxis, :]
            zi = encoder(xi)
            if i == 0:
                z = zi
            else:
                z = torch.cat((z, zi))
        #print('Z_SHAPE:', z.shape)

    else:
        z = encoder(x)

    z_ = z.detach().cpu()
    assignments = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete", affinity="precomputed"
                                          ).fit_predict(compute_similarity(z_, z_))
    centroids = torch.zeros((n_clusters, z_.shape[-1]))
    for cluster_ in range(n_clusters):
        index_cluster = [
            k for k, index in enumerate(assignments) if index == cluster_]
        centroids[cluster_] = torch.mean(z.detach()[index_cluster], dim=0)

    centroids = nn.Parameter(centroids)
    return centroids


class TemporalClustering(torch.nn.Module):
    '''
    References of this part
    Reference: https://github.com/HamzaG737/Deep-temporal-clustering/
    '''
    def __init__(self, encoder, n_clusters, centroids_init):
        super(TemporalClustering, self).__init__()
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.centroids = centroids_init

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def kl_loss_function(self, x_input, pred):
        out = x_input * torch.log((x_input) / (pred))
        return torch.mean(torch.sum(out, dim=1))

    def forward(self, x, varying=False):
        if varying:
            max_length = x.shape[-1]

            for i in range(x.shape[0]):
                # filter NAN
                lengths_i = max_length - torch.sum(
                    torch.isnan(x[i, :]), 1
                ).data.cpu().numpy()
                len_i = int(lengths_i)
                xi = x[i, :, :len_i]
                if len(xi.shape) < 3:
                    xi = xi[:, np.newaxis, :]
                zi = self.encoder(xi)
                if i == 0:
                    z = zi
                else:
                    z = torch.cat((z, zi))

        else:
            z = self.encoder(x)

        dist = compute_similarity(z, self.centroids)

        # q distribution
        Q = torch.pow((1 + dist), -1)
        Q = Q/torch.sum(Q, dim=1).view(-1, 1)

        # p distribution
        P = self.target_distribution(Q)
        p_q_mean = 1/2 * (P+Q)
        JS_loss = self.kl_loss_function(P, p_q_mean) + self.kl_loss_function(Q, p_q_mean)
        return JS_loss



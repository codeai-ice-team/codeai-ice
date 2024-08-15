import torch
from torch import nn
from torch.nn import functional as F
from pandas import DataFrame, Series

from ice.anomaly_detection.models.base import BaseAnomalyDetection


class GCLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int
            ):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
    
    def forward(self, adj, x):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        x = self.dense(x)
        norm = adj.sum(1)**(-1/2)
        x = norm[None, :] * adj * norm[:, None] @ x

        return x


class Directed_A(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int,
            alpha: float,
            k: int
            ):
        super().__init__()
        self.alpha = alpha
        self.k = k

        self.e1 = nn.Embedding(num_sensors, window_size)
        self.e2 = nn.Embedding(num_sensors, window_size)
        self.l1 = nn.Linear(window_size,window_size)
        self.l2 = nn.Linear(window_size,window_size)
    
    def forward(self, idx):
        m1 = torch.tanh(self.alpha*self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha*self.l2(self.e2(idx)))
        adj = F.relu(torch.tanh(self.alpha*torch.mm(m1, m2.transpose(1,0))))
        
        if self.k:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(idx.device)
            mask.fill_(float('0'))
            s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
            mask.scatter_(1,t1,s1.fill_(1))
            adj = adj*mask
        
        return adj


class GNNEncoder(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int,
            alpha: float,
            k: int
            ):
        super().__init__()
        self.idx = torch.arange(num_sensors)
        self.gcl1 = GCLayer(window_size, window_size // 2)
        self.gcl2 = GCLayer(window_size // 2, window_size // 8)
        self.A = Directed_A(num_sensors, window_size, alpha, k)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        adj = self.A(self.idx.to(x.device))
        x = self.gcl1(adj, x).relu()
        x = self.gcl2(adj, x).relu()
        return x
    

class Decoder(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int
            ):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.decoder = nn.Sequential(
            nn.Linear(window_size // 8 * num_sensors, window_size // 2 * num_sensors),
            nn.ReLU(),
            nn.Linear(window_size // 2 * num_sensors, num_sensors * window_size)
        )
    
    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.decoder(x)

        return x.view(-1, self.window_size, self.num_sensors)


class GSL_GNN(BaseAnomalyDetection):
    """
    GNN autoencoder consists of encoder with graph convolutional layers 
    and MLP decoder parts. The graph describing the data is constructed 
    during the training process using trainable parameters.
    """
    def __init__(
            self,
            window_size: int,
            stride: int = 1,
            batch_size: int = 128,
            lr: float = 0.001,
            num_epochs: int = 10,
            device: str = 'cpu',
            verbose: bool = False,
            name: str = 'gnn_anomaly_detection',
            random_seed: int = 42,
            val_ratio: float = 0.15,
            save_checkpoints: bool = False,
            threshold_level: float = 0.95,
            alpha: float = 0.2,
            k: int = None
        ):
        """
        Args:
            window_size (int): The window size to train the model.
            stride (int): The time interval between first points of consecutive 
                sliding windows in training.
            batch_size (int): The batch size to train the model.
            lr (float): The larning rate to train the model.
            num_epochs (float): The number of epochs to train the model.
            device (str): The name of a device to train the model. `cpu` and 
                `cuda` are possible.
            verbose (bool): If true, show the progress bar in training.
            name (str): The name of the model for artifact storing.
            random_seed (int): Seed for random number generation to ensure reproducible results.
            val_ratio (float): Proportion of the dataset used for validation, between 0 and 1.
            save_checkpoints (bool): If true, store checkpoints.
            threshold_level (float): Takes a value from 0 to 1. It specifies
                the quantile in the distribution of errors on the training
                dataset at which the threshold value is set.
            alpha (float): Saturation rate for adjacency matrix.
            k (int): Limit on the number of edges in the adjacency matrix.
        """
        super().__init__(
            window_size, stride, batch_size, lr, num_epochs, device, verbose, name, random_seed, 
            val_ratio, save_checkpoints, threshold_level
        )
        self.alpha = alpha
        self.k = k
    
    _param_conf_map = dict(BaseAnomalyDetection._param_conf_map,
            **{
                "alpha": ["MODEL", "ALPHA"]
            }
        )

    def _create_model(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            GNNEncoder(
                input_dim,
                self.window_size,
                self.alpha,
                self.k
                ),
            Decoder(input_dim, self.window_size)
            )

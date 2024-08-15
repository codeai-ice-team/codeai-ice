import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv

from ice.anomaly_detection.models.base import BaseAnomalyDetection


"""
The code of Stgat-Mad is taken from:
https://github.com/wagner-d/TimeSeAD
"""


class InputLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """
    def __init__(self, n_features, kernel_size=7):
        super(InputLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class StgatBlock(nn.Module):
    def __init__(self, n_features, window_size, dropout, embed_dim=None):
        super(StgatBlock, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else n_features

        self.embed_dim *= 2

        self.feature_gat_layers = GATConv(window_size, window_size)
        self.temporal_gat_layers = GATConv(n_features, n_features)

        self.temporal_gcn_layers = GCNConv(n_features, n_features)

    def forward(self, data, fc_edge_index, tc_edge_index):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        x = data.clone().detach()
        x = x.permute(0, 2, 1)
        batch_num, node_num, all_feature = x.shape

        x = x.reshape(-1, all_feature).contiguous()
        f_out = self.feature_gat_layers(x, fc_edge_index)
        f_out = F.relu(f_out)
        f_out = f_out.view(batch_num, node_num, -1)
        f_out = f_out.permute(0, 2, 1)
        z = f_out.reshape(-1, node_num).contiguous()

        t_out = self.temporal_gcn_layers(z, tc_edge_index)
        t_out = F.relu(t_out)
        t_out = t_out.view(batch_num, node_num, -1)

        return t_out.permute(0, 2, 1)


class BiLSTMLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(BiLSTMLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.bilstm = nn.LSTM(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, x):
        out, h = self.bilstm(x)
        out = out.permute(1,0,2)[-1, :, :] # Extracting from last layer
        return out


class BiLSTMDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(BiLSTMDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.bilstm = nn.LSTM(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, x):
        decoder_out, _ = self.bilstm(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = BiLSTMDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(2 * hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

# graph is 'fully-connect'
def get_fc_graph_struc(n_features):
    edge_indices = torch.tensor([[i, j] for j in range(n_features) for i in range(n_features) if i != j])
    return edge_indices.T.contiguous()


def get_tc_graph_struc(temporal_len):
    edge_indices = torch.tensor([[i, j] for j in range(temporal_len) for i in range(j)])
    return edge_indices.T.contiguous()


class STGAT(nn.Module):
    def __init__(
        self,
        n_features,
        window_size,
        embed_dim,
        layer_numb,
        lstm_n_layers,
        lstm_hid_dim,
        recon_n_layers,
        recon_hid_dim,
        dropout
    ):
        super(STGAT, self).__init__()

        layers1 = []
        layers2 = []
        layers3 = []

        self.layer_numb = layer_numb
        self.h_temp = []

        self.input_1 = InputLayer(n_features, 1)
        self.input_2 = InputLayer(n_features, 5)
        self.input_3 = InputLayer(n_features, 7)

        for i in range(layer_numb):
            layers1 += [StgatBlock(n_features, window_size, dropout, embed_dim)]
        for i in range(layer_numb):
            layers2 += [StgatBlock(n_features, window_size, dropout, embed_dim)]
        for i in range(layer_numb):
            layers3 += [StgatBlock(n_features, window_size, dropout, embed_dim)]

        self.stgat_1 = nn.Sequential(*layers1)
        self.stgat_2 = nn.Sequential(*layers2)
        self.stgat_3 = nn.Sequential(*layers3)

        self.bilstm = BiLSTMLayer(n_features * 3, lstm_hid_dim, lstm_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, 2 * lstm_hid_dim, recon_hid_dim, n_features, recon_n_layers, dropout)

        # Register as buffers so that tensors are moved to the correct device along with the rest of the model
        self.register_buffer('fc_edge_index', get_fc_graph_struc(n_features), persistent=False)
        self.register_buffer('tc_edge_index', get_tc_graph_struc(window_size), persistent=False)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        fc_edge_index_sets = get_batch_edge_index(self.fc_edge_index, x.shape[0], x.shape[2])
        tc_edge_index_sets = get_batch_edge_index(self.tc_edge_index, x.shape[0], x.shape[1])

        x_1 = x
        x_2 = self.input_2(x)
        x_3 = self.input_3(x)

        for layer in range(self.layer_numb):
            if layer==0:
                h_cat_1 = x_1 + self.stgat_1[layer](x_1, fc_edge_index_sets, tc_edge_index_sets)
                h_cat_2 = x_2 + self.stgat_2[layer](x_2, fc_edge_index_sets, tc_edge_index_sets)
                h_cat_3 = x_3 + self.stgat_3[layer](x_3, fc_edge_index_sets, tc_edge_index_sets)
            else:
                h_cat_1 = h_cat_1 + self.stgat_1[layer](h_cat_1, fc_edge_index_sets, tc_edge_index_sets)
                h_cat_2 = h_cat_2 + self.stgat_2[layer](h_cat_2, fc_edge_index_sets, tc_edge_index_sets)
                h_cat_3 = h_cat_3 + self.stgat_3[layer](h_cat_3, fc_edge_index_sets, tc_edge_index_sets)

        h_cat = torch.cat([h_cat_1, h_cat_2, h_cat_3], dim=2)

        out_end = self.bilstm(h_cat)
        h_end = out_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        recons = self.recon_model(h_end)

        return recons


class STGAT_MAD(BaseAnomalyDetection):
    """
    Stgat-Mad was presented at ICASSP 2022: "Stgat-Mad : Spatial-Temporal Graph 
    Attention Network For Multivariate Time Series Anomaly Detection".
    https://ieeexplore.ieee.org/abstract/document/9747274/
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
            name: str = 'stgat_anomaly_detection',
            random_seed: int = 42,
            val_ratio: float = 0.15,
            save_checkpoints: bool = False,
            threshold_level: float = 0.95,
            embed_dim: int = None,
            layer_numb: int = 2,
            lstm_n_layers: int = 1,
            lstm_hid_dim: int = 150,
            recon_n_layers: int = 1,
            recon_hid_dim: int = 150,
            dropout: float = 0.2
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
            embed_dim (int) : Embedding dimension.
            layer_numb (int) : Number of layers.
            lstm_n_layers (int) : Number of LSTM layers.
            lstm_hid_dim (int) : Hidden dimension of LSTM layers.
            recon_n_layers (int) : Number of reconstruction layers.
            recon_hid_dim (int) : Hidden dimension of reconstruction layers.
            dropout (float) : The rate of dropout.
        """
        super().__init__(
            window_size, stride, batch_size, lr, num_epochs, device, verbose, name, random_seed, 
            val_ratio, save_checkpoints, threshold_level
        )
        self.embed_dim = embed_dim
        self.layer_numb = layer_numb
        self.lstm_n_layers = lstm_n_layers
        self.lstm_hid_dim = lstm_hid_dim
        self.recon_n_layers = recon_n_layers
        self.recon_hid_dim = recon_hid_dim
        self.dropout = dropout

    _param_conf_map = dict(BaseAnomalyDetection._param_conf_map,
            **{
                "layer_numb": ["MODEL", "LAYER_NUMB"],
                "lstm_n_layers": ["MODEL", "LSTM_N_LAYERS"],
                "lstm_hid_dim": ["MODEL", "LSTM_HID_DIM"],
                "recon_n_layers": ["MODEL", "RECON_N_LAYERS"],
                "recon_hid_dim": ["MODEL", "RECON_HID_DIM"],
                "dropout": ["MODEL", "DROPOUT"]
            }
        )

    def _create_model(self, input_dim: int, output_dim: int):
        self.model = STGAT(
            n_features=input_dim,
            window_size=self.window_size,
            embed_dim=self.embed_dim,
            layer_numb=self.layer_numb,
            lstm_n_layers=self.lstm_n_layers,
            lstm_hid_dim=self.lstm_hid_dim,
            recon_n_layers=self.recon_n_layers,
            recon_hid_dim=self.recon_hid_dim,
            dropout=self.dropout
            )

from torch import nn
import torch
from ice.health_index_estimation.models.base import BaseHealthIndexEstimation
from pandas import DataFrame, Series
from torch.nn import functional as F


class LocalFeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(LocalFeatureExtraction, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.pool = nn.MaxPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim, timestep_dim):
        super(AttentionMechanism, self).__init__()
        self.feature_fc = nn.Linear(feature_dim, feature_dim)
        self.timestep_fc = nn.Linear(timestep_dim, timestep_dim)

    def forward(self, x):
        # Compute feature weights
        feature_weights = F.softmax(self.feature_fc(x), dim=-1)
        # Compute timestep weights
        timestep_weights = F.softmax(
            self.timestep_fc(x.transpose(1, 2)), dim=-1
        ).transpose(1, 2)

        # Apply attention
        feature_attention = feature_weights * x
        timestep_attention = timestep_weights * x
        return feature_attention * timestep_attention


class ParallelCNNBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        cnn_out_channels,
        cnn_kernel_size,
        lstm_hidden_size,
        lstm_num_layers,
        num_heads,
        sequence_length,
    ):
        super(ParallelCNNBiLSTM, self).__init__()
        self.attention = AttentionMechanism(input_dim, sequence_length)
        self.cnn1 = LocalFeatureExtraction(input_dim, cnn_out_channels, cnn_kernel_size)
        self.cnn2 = LocalFeatureExtraction(input_dim, cnn_out_channels, cnn_kernel_size)
        self.cnn3 = LocalFeatureExtraction(
            cnn_out_channels, cnn_out_channels, cnn_kernel_size
        )
        self.bilstm = nn.LSTM(
            cnn_out_channels * 2,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(
            sequence_length
        )  # Adaptive pooling to match dimensions
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.attention(x)  # Apply attention mechanism to x
        cnn1_out = self.cnn1(x.transpose(1, 2)).transpose(
            1, 2
        )  # Apply CNN and keep the original sequence length
        cnn2_out = self.cnn2(x.transpose(1, 2)).transpose(1, 2)

        cnn1_out = self.dropout(cnn1_out)
        cnn2_out = self.dropout(cnn2_out)

        cnn2_out = self.cnn3(cnn2_out.transpose(1, 2)).transpose(1, 2)

        # concatenate and pass to lstm
        combined_out = torch.cat((cnn1_out, cnn2_out), dim=-1)

        lstm_out, _ = self.bilstm(combined_out)

        combined_out = self.dropout(combined_out)
        return combined_out


class ToolWearMonitoringModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        window_size,
        cnn_out_channels=8,
        cnn_kernel_size=3,
        lstm_hidden_size=8,
        lstm_num_layers=2,
        num_heads=2,
    ):
        super(ToolWearMonitoringModel, self).__init__()
        self.parallel_cnn_bilstm = ParallelCNNBiLSTM(
            input_dim,
            cnn_out_channels,
            cnn_kernel_size,
            lstm_hidden_size,
            lstm_num_layers,
            num_heads,
            window_size,
        )
        self.linear = nn.Linear(
            lstm_hidden_size * window_size * 2, output_dim
        )  # Flatten before the linear layer

    def forward(self, x):
        x = self.parallel_cnn_bilstm(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.linear(x)
        return torch.squeeze(x, dim=1)


class TCM(BaseHealthIndexEstimation):
    """
    Wear condition monitoring module from https://doi.org/10.1016/j.jmsy.2021.12.002
    """

    def __init__(
        self,
        window_size: int = 64,
        stride: int = 300,
        batch_size: int = 64,
        lr: float = 5e-5,
        num_epochs: int = 50,
        device: str = "cpu",
        verbose: bool = True,
        name: str = "TCM_hi_estimation",
        random_seed: int = 42,
        val_ratio: float = 0.15,
        save_checkpoints: bool = False,
        lstm_hidden_size: int = 8,
        lstm_num_layers: int = 2,
    ):
        """
        Args:
            window_size (int): The window size to train the model.
            stride (int): The time interval between first points of consecutive sliding windows.
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
        """
        super().__init__(
            window_size,
            stride,
            batch_size,
            lr,
            num_epochs,
            device,
            verbose,
            name,
            random_seed,
            val_ratio,
            save_checkpoints,
        )
        self.val_metrics = True
        self.window_size = window_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

    _param_conf_map = dict(
        BaseHealthIndexEstimation._param_conf_map,
        **{"lstm_hidden_size": ["MODEL", "HIDDEN_DIM"]},
        **{"lstm_num_layers": ["MODEL", "NUM_LAYERS"]},
    )

    def _create_model(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            ToolWearMonitoringModel(
                input_dim=input_dim,
                output_dim=output_dim,
                window_size=self.window_size,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers
            )
        )

from torch import nn
import torch
from ice.health_index_estimation.models.base import BaseHealthIndexEstimation
from pandas import DataFrame, Series
from torch.nn import functional as F


class LSTMBlock(nn.Module):
    def __init__(self, input_dim, window_size, num_layers, dropout=0.2):
        super(LSTMBlock, self).__init__()
        self.lstm1a = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm1b = nn.LSTM(
            input_size=window_size,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.proj = nn.Linear(window_size + input_dim, window_size)

    def forward(self, x):
        lstm_out1a, _ = self.lstm1a(x)
        lstm_out1b, _ = self.lstm1b(
            x.transpose(1, 2)
        )  # Transpose input for the second LSTM

        lstm_out1 = torch.concat((lstm_out1a, lstm_out1b), dim=1)
        lstm_out1 = self.proj(lstm_out1.transpose(1, 2)).transpose(1, 2)
        skip_out = lstm_out1 + x
        return skip_out


class ImprovedModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        window_size=64,
        num_heads=1,
        num_layers=2,
        dropout=0.2,
    ):
        super(ImprovedModel, self).__init__()

        self.window_size = window_size
        self.input_dim = input_dim
        num_layers = num_layers

        # Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.lstm_block1 = LSTMBlock(self.input_dim, self.window_size, num_layers)
        self.lstm_block2 = LSTMBlock(self.input_dim, self.window_size, num_layers)
        # Linear transformation to align dimensions

        # Pooling Layer
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Attention Mechanism
        x, _ = self.attention(x, x, x)

        # First LSTM Layer
        x = self.lstm_block1(x)
        x = self.lstm_block2(x)

        # Pooling Layer
        pooled_out = self.pooling(x.permute(0, 2, 1)).squeeze(-1)

        # Fully Connected Layer
        output = self.fc(pooled_out)

        return torch.squeeze(output, dim=1)


class Stacked_LSTM(BaseHealthIndexEstimation):
    """
    Wear condition monitoring module from https://doi.org/10.1016/j.jmsy.2021.12.002
    """

    def __init__(
        self,
        window_size: int = 1024,
        stride: int = 300,
        batch_size: int = 64,
        lr: float = 5e-5,
        num_epochs: int = 50,
        device: str = "cpu",
        verbose: bool = True,
        name: str = "HI_stacked_LSTM",
        random_seed: int = 42,
        val_ratio: float = 0.15,
        save_checkpoints: bool = False,
        num_layers: int = 2,
        dropout_value: float = 0.2
    ):
        """
        Args:
            window_size (int): The window size to train the model.
            stride (int): The time interval between first points of consecutive sliding windows.
            hidden_dim (int): The dimensionality of the hidden layer in MLP.
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
        self.num_layers = num_layers
        self.dropout_value = dropout_value

    _param_conf_map = dict(
        BaseHealthIndexEstimation._param_conf_map,
        **{"num_layers": ["MODEL", "NUM_LAYERS"]},
        **{"dropout_value": ["MODEL", "DROPOUT"]},
    )

    def _create_model(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            ImprovedModel(
                input_dim=input_dim,
                output_dim=output_dim,
                window_size=self.window_size,
                num_layers=self.num_layers,
                dropout=self.dropout_value
            )
        )

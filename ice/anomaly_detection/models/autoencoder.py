import pandas as pd
from torch import nn

from ice.anomaly_detection.models.base import BaseAnomalyDetection


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            window_size: int,
            hidden_dims: list,
            decoder: bool = False,
            ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.hidden_dims = [input_dim * window_size]
        self.decoder = decoder
        if self.decoder:
            self.hidden_dims = hidden_dims + self.hidden_dims
        else:
            self.hidden_dims = self.hidden_dims + hidden_dims
        self.mlp = nn.Sequential(nn.Flatten())
        for i in range(len(hidden_dims)):
            self.mlp.append(nn.Linear(
                                self.hidden_dims[i],
                                self.hidden_dims[i + 1])
                                )
            if self.decoder and i + 1 == len(hidden_dims):
                break
            self.mlp.append(nn.ReLU())

    def forward(self, x):
        output = self.mlp(x)
        if self.decoder:
            return output.view(-1, self.window_size, self.input_dim)

        return output


class AutoEncoderMLP(BaseAnomalyDetection):
    """
    MLP autoencoder consists of MLP encoder and MLP decoder parts. Each
    sample is reshaped to a vector (B, L, C) -> (B, L * C) for calculations
    and to a vector (B, L * C) -> (B, L, C) for the output. Where B is the
    batch size, L is the sequence length, C is the number of sensors.
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
            name: str = 'ae_anomaly_detection',
            random_seed: int = 42,
            val_ratio: float = 0.15,
            save_checkpoints: bool = False,
            threshold_level: float = 0.95,
            hidden_dims: list = [256, 128, 64]
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
            hidden_dims (list): Dimensions of hidden layers in encoder/decoder.
        """
        super().__init__(
            window_size, stride, batch_size, lr, num_epochs, device, verbose, name, random_seed, 
            val_ratio, save_checkpoints, threshold_level
        )
        self.hidden_dims = hidden_dims

    _param_conf_map = dict(BaseAnomalyDetection._param_conf_map,
            **{
                "hidden_dims": ["MODEL", "HIDDEN_DIMS"]
            }
        )

    def _create_model(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            MLP(
                input_dim,
                self.window_size,
                hidden_dims=self.hidden_dims,
            ),
            MLP(
                input_dim,
                self.window_size,
                hidden_dims=self.hidden_dims[::-1],
                decoder=True
            )
        )

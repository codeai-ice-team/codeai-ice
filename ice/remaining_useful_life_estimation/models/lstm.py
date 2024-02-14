from torch import nn
from ice.remaining_useful_life_estimation.models.base import BaseRemainingUsefulLifeEstimation
from pandas import DataFrame, Series
import torch


class LSTM_model(nn.Module):
    def __init__(self, num_sensors, hidden_size, device, num_layers=2,):
        super(LSTM_model, self).__init__()
        self.num_sensors = num_sensors
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=num_sensors, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #internal state
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) 
        return output[:, -1, :]

class LSTM(BaseRemainingUsefulLifeEstimation):
    """
    Multilayer Perceptron (MLP) consists of input, hidden, output layers and
    ReLU activation. Each sample is reshaped to a vector (B, L, C) -> (B, L * C)
    where B is the batch size, L is the sequence length, C is the number of
    sensors.
    """

    def __init__(
        self,
        window_size: int = 32,
        stride: int = 1,
        hidden_dim: int = 512,
        hidden_size: int = 256,
        dropout_value: float = 0.5,
        batch_size: int = 64,
        lr: float = 1e-4,
        num_epochs: int = 35,
        device: str = "cpu",
        verbose: bool = True,
        name: str = "mlp_cmapss_rul",
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
        """
        super().__init__(
            window_size, stride, batch_size, lr, num_epochs, device, verbose, name
        )

        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.device = device
        self.dropout_value = dropout_value

        self.loss_array = []

    _param_conf_map = dict(
        BaseRemainingUsefulLifeEstimation._param_conf_map,
        **{"hidden_dim": ["MODEL", "HIDDEN_DIM"],
        "hidden_size": ["MODEL", "HIDDEN_SIZE"],
        "dropout_value": ["MODEL", "DROPOUT"],
        }
    )

    def _create_model(self, df: DataFrame, target: Series):
        num_sensors = df.shape[1] 

        self.model = nn.Sequential(
            nn.Dropout(self.dropout_value),
            LSTM_model(num_sensors, self.hidden_size, self.device),
            nn.Flatten(),
            nn.Linear(self.hidden_size, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_value),
            nn.Linear(self.hidden_dim, 1),
            nn.Flatten(start_dim=0),
        )

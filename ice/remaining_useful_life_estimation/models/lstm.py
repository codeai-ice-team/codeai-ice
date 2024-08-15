from torch import nn
from ice.remaining_useful_life_estimation.models.base import BaseRemainingUsefulLifeEstimation
from pandas import DataFrame, Series
import torch


class LSTM_model(nn.Module):
    """
    Long short-term memory (LSTM) is reccurent neural network type, 
    pytorch realisation https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html 
    """
    def __init__(
            self, 
            input_dim, 
            hidden_size=512, 
            device="cpu", 
            num_layers=2,):
        """
        Args:
            input_dim (int): The dimension size of input data, related to the sensor amount in industry probles.
            hidden_size (int): The number of features in the hidden state of the model.
            device (str): The name of a device to train the model. `cpu` and `cuda` are possible.
            num_layers (int): The number of stacked reccurent layers of the classic LSTM architecture.
        """
        super(LSTM_model, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #internal state
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) 
        return output[:, -1, :]

class LSTM(BaseRemainingUsefulLifeEstimation):
    """
    Long short-term memory (LSTM) model consists of the classical LSTM architecture stack and 
    two-layer MLP with SiLU nonlinearity and dropout to make the final prediction.
    
    Each sample is moved to LSTM and reshaped to a vector (B, L, C) -> (B, hidden_size, C)
    Then the sample is reshaped to a vector (B, hidden_size, C) -> (B, hidden_size * C)
    """

    def __init__(
        self,
        window_size: int = 32,
        stride: int = 1,
        hidden_dim: int = 512,
        hidden_size: int = 256, 
        num_layers: int =2,
        dropout_value: float = 0.5,
        batch_size: int = 64,
        lr: float = 1e-4,
        num_epochs: int = 35,
        device: str = "cpu",
        verbose: bool = True,
        name: str = "mlp_cmapss_rul",
        random_seed: int = 42,
        val_ratio: float = 0.15,
        save_checkpoints: bool = False
    ):
        """
        Args:
            window_size (int): The window size to train the model.
            stride (int): The time interval between first points of consecutive sliding windows.
            hidden_dim (int): The dimensionality of the hidden layer in MLP.
            hidden_size (int): The number of features in the hidden state of the model.
            num_layers (int): The number of stacked reccurent layers of the classic LSTM architecture.
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
            window_size, stride, batch_size, lr, num_epochs, device, verbose, name, random_seed, val_ratio, save_checkpoints
        )
        self.val_metrics = True

        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dropout_value = dropout_value

        self.loss_array = []

    _param_conf_map = dict(
        BaseRemainingUsefulLifeEstimation._param_conf_map,
        **{"hidden_dim": ["MODEL", "HIDDEN_DIM"],
        "hidden_size": ["MODEL", "HIDDEN_SIZE"],
        "num_layers": ["MODEL", "NUM_LAYERS"],
        "dropout_value": ["MODEL", "DROPOUT"],
        }
    )

    def _create_model(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            nn.Dropout(self.dropout_value),
            LSTM_model(input_dim, self.hidden_size, self.device, self.num_layers),
            nn.Flatten(),
            nn.Linear(self.hidden_size, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_value),
            nn.Linear(self.hidden_dim, 1),
            nn.Flatten(start_dim=0),
        )

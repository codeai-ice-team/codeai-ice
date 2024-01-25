from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from torch import nn

from ice.anomaly_detection.models.base import BaseAnomalyDetection


class MLP(nn.Module):
    def __init__(
            self,
            num_sensors: int,
            window_size: int,
            num_layers: int,
            hidden_dims: list,
            type: str
            ):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.hidden_dims = [num_sensors * window_size]
        self.type = type
        if hidden_dims is not None and self.type == 'encoder':
            self.hidden_dims = self.hidden_dims + hidden_dims
        elif hidden_dims is not None and self.type == 'decoder':
            self.hidden_dims = hidden_dims + self.hidden_dims
        else:
            for i in range(num_layers):
                dim = self.hidden_dims[i] // 2
                self.hidden_dims.append(dim)
            if self.type == 'decoder':
                self.hidden_dims.reverse()
        self.mlp = nn.Sequential(nn.Flatten())
        for i in range(num_layers):
            self.mlp.append(nn.Linear(
                                self.hidden_dims[i],
                                self.hidden_dims[i + 1])
                                )
            if self.type == 'decoder' and i + 1 == num_layers:
                break
            self.mlp.append(nn.ReLU())

    def forward(self, x):
        output = self.mlp(x)
        if self.type == 'decoder':
            return output.view(-1, self.window_size, self.num_sensors)

        return output


class AutoEncoderMLP(BaseAnomalyDetection):
    """
    Autoencoder (AE) consists of encoder and decoder parts. Each
    sample is reshaped to a vector (B, L, C) -> (B, L * C) for calculations
    and to a vector (B, L * C) -> (B, L, C) for the output. Where B is the
    batch size, L is the sequence length, C is the number of sensors.
    """
    def __init__(
            self,
            window_size: int,
            batch_size: int = 128,
            lr: float = 0.001,
            num_epochs: int = 10,
            device: str = 'cpu',
            verbose: bool = False,
            name: str = 'ae_anomaly_detection',
            threshold: float = 0.95,
            hidden_dims: list = [256, 128, 64],
            ):
        """
        Args:
            window_size (int): The window size to train the model.
            batch_size (int): The batch size to train the model.
            lr (float): The larning rate to train the model.
            num_epochs (float): The number of epochs to train the model.
            device (str): The name of a device to train the model. `cpu` and 
                `cuda` are possible.
            verbose (bool): If true, show the progress bar in training.
            name (str): The name of the model for artifact storing.
            threshold (float): The boundary for anomaly detection.
            hidden_dims (list): Dimensions of hidden layers in encoder/decoder.
        """
        super().__init__(
            window_size, batch_size, lr, num_epochs, device, verbose, name, threshold
        )

        self.window_size = window_size
        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.threshold = threshold
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.preprocessing = True
        self.scaler = StandardScaler()

    _param_conf_map = dict(BaseAnomalyDetection._param_conf_map,
            **{
                "hidden_dims": ["MODEL", "HIDDEN_DIMS"]
            }
        )

    def _create_model(self, df: DataFrame):
        num_sensors = df.shape[1]
        self.model = nn.Sequential(
            MLP(
                num_sensors,
                self.window_size,
                self.num_layers,
                hidden_dims=self.hidden_dims,
                type='encoder'
            ),
            MLP(
                num_sensors,
                self.window_size,
                self.num_layers,
                hidden_dims=self.hidden_dims[::-1],
                type='decoder'
            )
            )

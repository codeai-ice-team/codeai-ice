from torch import nn
from ice.fault_diagnosis.models.base import BaseFaultDiagnosis
from pandas import DataFrame, Series

class MLP(BaseFaultDiagnosis):
    """
    Multilayer Perceptron (MLP) consists of input, hidden, output layers and
    ReLU activation. Each sample is reshaped to a vector (B, L, C) -> (B, L * C)
    where B is the batch size, L is the sequence length, C is the number of 
    sensors.
    """
    def __init__(
            self, 
            window_size: int,
            hidden_dim: int=256,
            batch_size: int=128,
            lr: float=0.001,
            num_epochs: int=10,
            device: str='cpu',
            verbose: bool=False,
            name: str='mlp_fault_diagnosis'
        ):
        """
        Args:
            window_size (int): The window size to train the model.
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
            window_size, batch_size, lr, num_epochs, device, verbose, name
        )

        self.hidden_dim = hidden_dim

    _param_conf_map = dict(BaseFaultDiagnosis._param_conf_map, 
            **{
                "hidden_dim" : ["MODEL", "HIDDEN_DIM"]
            }
        )   

    def _create_model(self, df: DataFrame, target: Series):
        num_sensors = df.shape[1]
        num_classes = len(set(target))
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_sensors * self.window_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes),
        )

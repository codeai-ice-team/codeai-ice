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
            stride: int = 1,
            hidden_dim: int=256,
            batch_size: int=128,
            lr: float=0.001,
            num_epochs: int=10,
            device: str='cpu',
            verbose: bool=False,
            name: str='mlp_fault_diagnosis',
            random_seed: int = 42,
            val_ratio: float = 0.15,
            save_checkpoints: bool = False
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
            random_seed (int): Seed for random number generation to ensure reproducible results.
            val_ratio (float): Proportion of the dataset used for validation, between 0 and 1.
            save_checkpoints (bool): If true, store checkpoints.
        """
        super().__init__(
            window_size, stride, batch_size, lr, num_epochs, device, verbose, name, random_seed, val_ratio, save_checkpoints
        )
        self.val_metrics = True

        self.hidden_dim = hidden_dim

    _param_conf_map = dict(BaseFaultDiagnosis._param_conf_map, 
            **{
                "hidden_dim" : ["MODEL", "HIDDEN_DIM"]
            }
        )   

    def _create_model(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * self.window_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )

from torch import nn
from ice.remaining_useful_life_estimation.models.base import BaseRemainingUsefulLifeEstimation
from pandas import DataFrame, Series
import torch

class KLLoss:
    def __init__(self, z_dims=False):
        self.name = "KLLoss"
        self.z_dims = z_dims

    def __call__(self, mean, log_var):
        if self.z_dims:
            mean = mean[:, self.z_dims[0]: self.z_dims[1]]
            log_var = log_var[:, self.z_dims[0]: self.z_dims[1]]
        loss = (-0.5 * (1 + log_var - mean ** 2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        return loss


class RegLoss:
    def __init__(self, z_dims=2):
        self.name = "RegLoss"
        self.z_dims = z_dims
        self.criterion = nn.MSELoss()

    def __call__(self, y, y_hat):
        return self.criterion(y, y_hat)


class TotalLoss:
    def __init__(self):
        self.RegLoss = RegLoss()
        self.KLLoss = KLLoss()

    
    def __call__(self, model_output, y):

        if type(model_output) != torch.Tensor:
            
            y_hat, mean, log_var = model_output

            y = y.to(torch.float64)
            y_hat = y_hat.to(torch.float64)
            mean = mean.to(torch.float64)
            log_var = log_var.to(torch.float64)

            return self.RegLoss(y, y_hat) + self.KLLoss(mean, log_var)
        
        else:
            return self.RegLoss(y.to(torch.float64), model_output.to(torch.float64))


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_dim: int,
        dropout_lstm: float,
        num_layers: int,
        dropout: float=0,
        bidirectional=True,
    ):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.p_lstm = dropout_lstm
        self.p = dropout

        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.fc_mean = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * self.hidden_size, out_features=latent_dim
            ),
        )

        self.fc_log_var = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * self.hidden_size, out_features=latent_dim
            ),
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        batch_size = x.shape[0]
        _, (h_n, _) = self.lstm(x)
        
        h_n = h_n.view(
            self.num_layers, self.num_directions, batch_size, self.hidden_size
        )
        if self.bidirectional:
            h = torch.cat((h_n[-1, -2, :, :], h_n[-1, -1, :, :]), dim=1)
        else:
            h = h_n[-1, -1, :, :]
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )
        return z, mean, log_var



class RVAE_model(nn.Module):

    def __init__(
        self,
        window_size: int, 
        hidden_size: int,
        input_dim: int,
        num_layers: int,
        latent_dim: int,
        hidden_dim: int,
        dropout_value: float
    ):
        super(RVAE_model, self).__init__()
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size=input_dim,
        hidden_size=self.hidden_size,
        latent_dim=latent_dim,
        dropout_lstm=dropout_value,
        num_layers=num_layers
)
        self.p = dropout_value
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.p),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        y_hat = self.regressor(z)
        if self.training:
            return y_hat.flatten(start_dim=0), mean, log_var

        return y_hat.flatten(start_dim=0) 

class RVAE(BaseRemainingUsefulLifeEstimation):
    """
    Bidirectional LSTM for interpretable 2D latent space from paper https://doi.org/10.1016/j.ress.2022.108353.
    
    """

    def __init__(
        self,
        window_size: int = 32,
        stride: int = 1,
        hidden_size: int = 21, 
        hidden_dim: int = 200,
        latent_dim: int =2,
        num_layers: int =2,
        dropout_value: float = 0.5,
        batch_size: int = 256,
        lr: float = 1e-4,
        num_epochs: int = 35,
        device: str = "cpu",
        verbose: bool = True,
        name: str = "rvae_cmapss_rul",
        random_seed: int = 42,
        val_ratio: float = 0.15,
        save_checkpoints: bool = False
    ):
        """
        Args:
            window_size (int): The window size to train the model.
            stride (int): The time interval between first points of consecutive sliding windows.
            hidden_size (int): The number of features in the reccurent models hidden state.
            hidden_dim (int): The dimensionality of the hidden layer in MLP.
            num_layers (int): The number of stacked reccurent layers in the reccurent models.
            dropout_value (float): The dropout value for regularization
            batch_size (int): The batch size to train the model.
            lr (float): The larning rate to train the model.
            num_epochs (float): The number of epochs to train the model.
            device (str): The name of a device to train the model. `cpu` and
                `cuda` are possible.
            verbose (bool): If true, show the progress bar in training.
            name (str): The name of the model for artifact storing.
            save_checkpoints (bool): If true, store checkpoints.
        """
        super().__init__(
            window_size, stride, batch_size, lr, num_epochs, device, verbose, name, random_seed, val_ratio, save_checkpoints
        )


        self.device = device
        self.dropout_value = dropout_value
        self.window_size=window_size 
        self.hidden_size=hidden_size 
        self.num_layers=num_layers
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim

        self.loss_array = []

    _param_conf_map = dict(
        BaseRemainingUsefulLifeEstimation._param_conf_map,
        **{"hidden_dim": ["MODEL", "HIDDEN_DIM"],
        "hidden_size": ["MODEL", "HIDDEN_SIZE"],
        "num_layers": ["MODEL", "NUM_LAYERS"],
        "dropout_value": ["MODEL", "DROPOUT"],
        "latent_dim": ["MODEL", "LATENT_DIM"]
        }
    )


    def _prepare_for_training(self, input_dim: int, output_dim: int):
        self.loss_fn = TotalLoss()# nn.L1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def _create_model(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            RVAE_model(window_size=self.window_size, 
                        hidden_size=self.hidden_size, 
                        input_dim=input_dim,
                        num_layers=self.num_layers,
                        latent_dim=self.latent_dim,
                        hidden_dim=self.hidden_dim,
                        dropout_value=self.dropout_value
                            ),
            # nn.Flatten(start_dim=0),
        )

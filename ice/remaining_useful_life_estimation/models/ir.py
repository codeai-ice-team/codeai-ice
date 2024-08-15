from torch import nn
from ice.remaining_useful_life_estimation.models.base import (
    BaseRemainingUsefulLifeEstimation,
)
from pandas import DataFrame, Series
import torch
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=21, max_len=32):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]

class LSTMBlock(nn.Module):
    def __init__(self, input_dim, window_size, num_layers, device="cpu", dropout=0.5):
        super(LSTMBlock, self).__init__()
        self.device=device
        self.lstm1a = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm1b = nn.LSTM(
            input_size=window_size,
            hidden_size=window_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.proj = nn.Linear(window_size + input_dim, window_size)
        self.hidden_size = input_dim
        self.window_size = window_size
        self.num_layers=num_layers

    def forward(self, x):
        skip = x
        
        h_1a = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c_1a = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)

        
        h_1b = torch.zeros(self.num_layers, x.size()[0], self.window_size).to(self.device)
        c_1b = torch.zeros(self.num_layers, x.size()[0], self.window_size).to(self.device)

        lstm_out1a, _ = self.lstm1a(x, (h_1a, c_1a))
        lstm_out1b, _ = self.lstm1b(
            x.transpose(1, 2), (h_1b, c_1b)
        ) 

        lstm_out1a = lstm_out1a + x
        lstm_out1b = lstm_out1b.transpose(1, 2) + x

        lstm_out1 = (lstm_out1a + lstm_out1b) / 2 
        return lstm_out1


class Transformer_encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        dropout=0.2,
        output_dim=1,
        window_size=32,
        num_layers=1,
        num_heads=3,
        dim_feedforward=128,
        noise=0.5,
        device='cpu'
    ):
        super(Transformer_encoder, self).__init__()
        self.seq_len = window_size
        self.feature_dim = input_dim
        self.output_dim = output_dim
        self.device =device
        self.noise=noise

        if self.feature_dim % num_heads != 0:
            num_heads = 1

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            activation = nn.LeakyReLU(),
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, norm=torch.nn.LayerNorm(self.feature_dim))
        self.lstm_block1 = LSTMBlock(self.feature_dim, self.seq_len, num_layers=2, dropout=dropout, device=device)

        self.latent_layer = nn.Linear(self.seq_len * self.feature_dim * 2, self.output_dim)
        self.positional_embedding = PositionalEmbedding(self.feature_dim, self.seq_len)
        self.fc = [
            nn.Linear(self.seq_len * self.feature_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
        ]

        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        if self.training:
            x += torch.rand(x.shape).to(self.device) * self.noise
            
        x = x + self.positional_embedding(x)
        encoded = self.encoder(x)

        encoded = encoded + x
        encoded = encoded.flatten(start_dim=1)

        lstmed = self.lstm_block1(x)
        lstmed = lstmed.flatten(start_dim=1)

        concat = torch.concat((encoded, lstmed), dim=1)

        latent = self.latent_layer(concat)
        

        return torch.squeeze(latent, dim=1)




class IR(BaseRemainingUsefulLifeEstimation):
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
        noise: float = 0.012097363825546333,
        num_layers: int = 3,
        dropout_value: float = 0.2,
        batch_size: int = 256,
        lr: float = 0.004091793998895119,
        num_epochs: int = 27,
        device: str = "cpu",
        verbose: bool = True,
        name: str = "ir_model",
        random_seed: int = 42,
        val_ratio: float = 0.15,
        save_checkpoints: bool = False,
        dim_feedforward: int = 154
    ):
        """
        Args:
            window_size (int): The window size to train the model.
            stride (int): The time interval between first points of consecutive sliding windows.
            noise (float): The input noise for model training
            hidden_dim (int): The dimensionality of the hidden layer in MLP.
            hidden_size (int): The number of features in the hidden state of the model.
            dropout_value (float): Dropout probability in model layers
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
            dim_feedforward (int): The dimension of feedforward transformer encoder layer
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

        self.num_layers = num_layers
        self.device = device
        self.dropout_value = dropout_value
        self.window_size = window_size
        self.noise = noise
        self.dim_feedforward=dim_feedforward

        self.loss_array = []

    _param_conf_map = dict(
        BaseRemainingUsefulLifeEstimation._param_conf_map,
        **{
            "num_layers": ["MODEL", "NUM_LAYERS"],
            "noise": ["MODEL", "NOISE"],
            "dim_feedforward": ["MODEL", "DIM_FEEDFORWARD"],
            "dropout_value": ["MODEL", "DROPOUT"],
        }
    )

    def _create_model(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            Transformer_encoder(
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=self.num_layers,
                device=self.device,
                noise=self.noise,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_value
            )
        )

from torch import nn
import torch
from ice.health_index_estimation.models.base import BaseHealthIndexEstimation
from pandas import DataFrame, Series
from torch.nn import functional as F
import math


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm(attn_output + x)  # Add & Norm
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(PositionwiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(ff_dim, embed_dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.permute(
            0, 2, 1
        )  
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = x.permute(
            0, 2, 1
        )   # Add & Norm
        return self.drop(x)


class DistillationLayer(nn.Module):
    def __init__(self, input_dim, distilled_dim=18, kernel_size1=3, kernel_size2=3):
        super(DistillationLayer, self).__init__()
        self.conv1 = nn.Conv1d(
            input_dim, distilled_dim, kernel_size=kernel_size1, stride=1, padding=1
        )
        self.batch_norm = nn.BatchNorm1d(distilled_dim)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv1d(
            distilled_dim, distilled_dim, kernel_size=kernel_size2, stride=2, padding=1
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = x.permute(
            0, 2, 1
        )  
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = x.permute(
            0, 2, 1
        )  
        return x


class Basic_block(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(Basic_block, self).__init__()
        self.multihead_attn = MultiHeadAttentionLayer(embed_dim, num_heads)
        self.ffn = PositionwiseFeedForward(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dist = DistillationLayer(embed_dim, embed_dim)

    def forward(self, x):
        x = self.multihead_attn(x)
        x = self.dropout(x) + x
        x = self.layer_norm(x)

        x = self.dropout(self.ffn(x)) + x

        x = self.layer_norm(x)
        x = self.dist(x)
        return x


class InformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(InformerEncoder, self).__init__()
        self.Basic_block_1 = Basic_block(embed_dim, num_heads, ff_dim, dropout)
        self.Basic_block_2 = Basic_block(embed_dim, num_heads, ff_dim, dropout)

    def forward(self, x):
        x = self.Basic_block_1(x)
        x = self.Basic_block_2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, feature_size, max_len=1500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, feature_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_size, 2).float()
            * (-math.log(10000.0) / feature_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[: feature_size // 2]
        )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), : x.size(2)]
        return x


class StackedBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(StackedBiGRU, self).__init__()
        self.bigru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        x, _ = self.bigru(x)
        return x


class IE_SBiGRU_Model(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=64,
        num_heads=8,
        ff_dim=256,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
    ):
        super(IE_SBiGRU_Model, self).__init__()
        self.input_proj = nn.Linear(
            input_dim, embed_dim
        )  # Projection to embedding dimension

        self.pos_encoder = PositionalEncoding(embed_dim)

        self.encoder = InformerEncoder(embed_dim, num_heads, ff_dim, dropout)
        self.bigru = StackedBiGRU(embed_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # Output Layer
        self.tanh = nn.Tanh()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # Global Max Pooling

    def forward(self, x):
        x = self.input_proj(x)  # Project input to embedding dimension
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.bigru(x)
        x = self.global_max_pool(x.permute(0, 2, 1)).squeeze(-1)  # Permute for pooling
        x = self.tanh(x)
        x = self.fc(x)
        return torch.squeeze(x, dim=1)


class IE_SBiGRU(BaseHealthIndexEstimation):
    """
    Wear condition monitoring model from https://doi.org/10.1016/j.rcim.2022.102368 SOTA paper
    """

    def __init__(
        self,
        window_size: int = 1024,
        num_layers: int = 3,
        hidden_dim: int = 128,
        ff_dim: int = 256,
        stride: int = 300,
        batch_size: int = 64,
        lr: float = 0.0031789041005068647,
        num_epochs: int = 55,
        device: str = "cpu",
        verbose: bool = True,
        name: str = "IE_SBiGRU_hi_estimation",
        random_seed: int = 42,
        val_ratio: float = 0.15,
        save_checkpoints: bool = False,
    ):
        """
        Args:
            window_size (int): The window size to train the model.
            stride (int): The time interval between first points of consecutive sliding windows.
            num_layers (int): The amount of BiGRU layers in the model.
            hidden_dim (int): The regression head hidden linear layer size.
            ff_dim (int): The CNN projection dim in feedforward module of Informer encoder.
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
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim

    _param_conf_map = dict(
        BaseHealthIndexEstimation._param_conf_map,
        **{"num_layers": ["MODEL", "NUM_LAYERS"]},
        **{"hidden_dim": ["MODEL", "HIDDEN_DIM"]},
        **{"ff_dim": ["MODEL", "FF_DIM"]}
    )

    def _create_model(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            IE_SBiGRU_Model(
                input_dim=input_dim,
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                ff_dim=self.ff_dim,
            )
        )

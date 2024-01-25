from torch import nn
from torch.nn import functional as F
from ice.fault_diagnosis.models.base import BaseFaultDiagnosis
from pandas import DataFrame, Series


"""
The implementation of _ResidualBlock is taken from the Darts library:
https://github.com/unit8co/darts
"""
class _ResidualBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_fn,
        nr_blocks_below: int,
        num_layers: int,
        input_size: int,
        target_size: int,
    ):
        super().__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim,
            num_filters,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        self.conv2 = nn.Conv1d(
            num_filters,
            output_dim,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        residual = x

        # first step
        left_padding = (self.dilation_base**self.nr_blocks_below) * (
            self.kernel_size - 1
        )
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(F.relu(self.conv1(x)))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)

        # add residual
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual

        return x


class _TCNModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        kernel_size: int,
        hidden_dim: int,
        num_layers: int,
        dilation_base: int,
        output_dim: int,
        dropout: float,
        seq_len: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = _ResidualBlock(
                hidden_dim,
                kernel_size,
                dilation_base,
                nn.Dropout(p=dropout),
                i,
                num_layers,
                input_dim,
                hidden_dim,
            )
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)
        self.projection_head = nn.Linear(hidden_dim * seq_len, output_dim)

    def forward(self, x):
        # data is of size (batch_size, input_chunk_length, input_size)
        x = x.transpose(1, 2)
        for res_block in self.res_blocks_list:
            x = res_block(x)
        x = x.transpose(1, 2)
        x = x.reshape(-1, self.hidden_dim * self.seq_len)
        x = self.projection_head(x)
        return x


class TCN(BaseFaultDiagnosis):
    """
    Temporal Convolutional Network (TCN)-based Fault Diagnosis method. The 
    implementation is based on the paper Lomov, Ildar, et al. "Fault detection 
    in Tennessee Eastman process with temporal deep learning models." Journal 
    of Industrial Information Integration 23 (2021): 100216.
    """
    def __init__(
            self, 
            window_size: int,
            hidden_dim: int=256,
            kernel_size: int=5,
            num_layers: int=4,
            dilation_base: int=2,
            dropout: float=0.2,
            batch_size: int=128,
            lr: float=0.001,
            num_epochs: int=10,
            device: str='cpu',
            verbose: bool=False,
            name: str='tcn_fault_diagnosis'
        ):
        """
        Args:
            window_size (int): The window size to train the model.
            hidden_dim (int): The number of channels in the hidden layers of TCN.
            kernel_size (int): The kernel size of the residual blocks of TCN.
            num_layers (int): The number of residual blocks in TCN.
            dilation_base (int): The base of the exponent that will determine the dilation on every layer.
            dropout (float): The rate of dropout in training of TCN.
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
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dilation_base = dilation_base
        self.dropout = dropout

    _param_conf_map = dict(BaseFaultDiagnosis._param_conf_map, 
            **{
                "hidden_dim" : ["MODEL", "HIDDEN_DIM"],
                "kernel_size" : ["MODEL", "KERNEL_SIZE"],
                "num_layers" : ["MODEL", "NUM_LAYERS"],
                "dilation_base" : ["MODEL", "DILATION_BASE"],
                "dropout" : ["MODEL", "DROPOUT"],
            }
        )   

    def _create_model(self, df: DataFrame, target: Series):
        num_sensors = df.shape[1]
        num_classes = len(set(target))
        self.model = _TCNModule(
            input_dim=num_sensors,
            kernel_size=self.kernel_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dilation_base=self.dilation_base,
            output_dim=num_classes,
            dropout=self.dropout,
            seq_len=self.window_size,
        )

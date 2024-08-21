from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import optuna

from ice.base import BaseModel, SlidingWindowDataset
from ice.anomaly_detection.metrics import (
    accuracy, true_positive_rate, false_positive_rate)


class BaseAnomalyDetection(BaseModel, ABC):
    """Base class for all anomaly detection models."""

    @abstractmethod
    def __init__(
            self,
            window_size: int,
            stride: int,
            batch_size: int,
            lr: float,
            num_epochs: int,
            device: str,
            verbose: bool,
            name: str,
            random_seed: int,
            val_ratio: float,
            save_checkpoints: bool,
            threshold_level: float = 0.95
            ):
        """
        Args:
            window_size (int): The window size to train the model.
            stride (int): The time interval between first points of consecutive 
                sliding windows in training.
            batch_size (int): The batch size to train the model.
            lr (float): The learning rate to train the model.
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
        """
        super().__init__(window_size, stride, batch_size, lr, num_epochs, device, verbose, name, random_seed, val_ratio, save_checkpoints)
        self.val_metrics = False

        self.threshold_level = threshold_level
        self.threshold_value = None

    def fit(self, df: pd.DataFrame, target: pd.Series = None,
            epochs: int = None, save_path: str = None, trial: optuna.Trial = None,
        force_model_ctreation: bool = False):
        """Fit (train) the model by a given dataset.

        Args:
            df (pandas.DataFrame): A dataframe with sensor data. Index has 
                two columns: `run_id` and `sample`. All other columns a value of 
                sensors.
            target (pandas.Series): A series with target values. Indes has two
                columns: `run_id` and `sample`. It is omitted for anomaly
                detection task.
            epochs (int): The number of epochs for training step. If None, 
                self.num_epochs parameter is used.
            save_path (str): Path to save checkpoints. If None, the path is
                created automatically.
        """
        if trial:
            super().fit(df, target, epochs, save_path, trial=trial, force_model_ctreation=True)
        else:
            super().fit(df, target, epochs, save_path)

        error = []
        for sample, target in tqdm(
            self.dataloader, desc='Steps ...', leave=False, disable=(not self.verbose)
        ):
            sample = sample.to(self.device)
            with torch.no_grad():
                pred = self.model(sample)
            error.append(self.loss_no_reduction(pred, sample).mean(dim=(1, 2)))
        error = torch.concat(error)
        self.threshold_value = torch.quantile(error, self.threshold_level).item()
        if self.save_checkpoints:
                self.save_checkpoint(save_path)

    _param_conf_map = dict(BaseModel._param_conf_map,
            **{
                "threshold_level": ["MODEL", "THRESHOLD_LEVEL"]
            }
        )
    
    def _prepare_for_training(self, input_dim: int, output_dim: int):
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.L1Loss()
        self.loss_no_reduction = nn.L1Loss(reduction='none')

    def _predict(self, sample: torch.Tensor) -> torch.Tensor:
        input = sample.to(self.device)
        output = self.model(input)
        error = self.loss_no_reduction(output, input).mean(dim=(1, 2))
        return (error > self.threshold_value).float().cpu()
    
    def _validate_inputs(self, df: pd.DataFrame, target: pd.Series):
        if target is not None:
            assert df.shape[0] == target.shape[0], f"target is incompatible with df by the length: {df.shape[0]} and {target.shape[0]}."
            assert np.all(df.index == target.index), "target's index and df's index are not the same."
        assert df.index.names == (['run_id', 'sample']), "An index should contain columns `run_id` and `sample`."            
        assert len(df) >= self.window_size, "window size is larger than the length of df."

    def _calculate_metrics(self, pred: torch.tensor, target: torch.tensor) -> dict:
        metrics = {
            'accuracy': accuracy(pred, target),
            'true_positive_rate': true_positive_rate(pred, target),
            'false_positive_rate': false_positive_rate(pred, target),
        }
        return metrics
    
    def _set_dims(self, df: pd.DataFrame, target: pd.Series):
        self.input_dim = df.shape[1]
        self.output_dim = 1

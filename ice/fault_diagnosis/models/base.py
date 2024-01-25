from abc import ABC, abstractmethod
import pandas as pd
from tqdm.auto import trange, tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ice.fault_diagnosis.utils import SlidingWindowDataset
from ice.base import BaseModel
from ice.fault_diagnosis.metrics import (
    accuracy, correct_daignosis_rate, true_positive_rate, false_positive_rate)

class BaseFaultDiagnosis(BaseModel, ABC):
    """Base class for all fault diagnosis models."""

    @abstractmethod
    def __init__(
            self,
            window_size: int,
            batch_size: int, 
            lr: float, 
            num_epochs: int, 
            device: str, 
            verbose: bool,
            name: str
        ):
        """
        Args:
            window_size (int): The window size to train the model.
            batch_size (int): The batch size to train the model.
            lr (float): The learning rate to train the model.
            num_epochs (float): The number of epochs to train the model.
            device (str): The name of a device to train the model. `cpu` and 
                `cuda` are possible.
            verbose (bool): If true, show the progress bar in training.
            name (str): The name of the model for artifact storing.
        """
        super().__init__(batch_size, lr, num_epochs, device, verbose, name)
        self._cfg.path_set(["TASK"], "fault_diagnosis")

        self.window_size = window_size
        self.loss_fn = None

    _param_conf_map = dict(BaseModel._param_conf_map,
            **{
                "window_size" : ["MODEL", "WINDOW_SIZE"]
            }
        )

    def _fit(self, df: pd.DataFrame, target: pd.Series):
        assert len(df) >= self.window_size, "window size is larger than the length of df."
        num_classes = len(set(target))
        weight = torch.ones(num_classes, device=self.device) * 0.5
        weight[1:] /= num_classes - 1
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        self._train_nn(df, target)

    def _predict(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.to(self.device)
        logits = self.model(sample)
        return logits.argmax(axis=1).cpu()

    def _train_nn(self, df: pd.DataFrame, target: pd.Series):
        self.model.train()
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        dataset = SlidingWindowDataset(df, target, window_size=self.window_size)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for e in trange(self.num_epochs, desc='Epochs ...', disable=(not self.verbose)):
            for sample, target in tqdm(self.dataloader, desc='Steps ...', leave=False, disable=(not self.verbose)):
                sample = sample.to(self.device)
                target = target.to(self.device)
                logits = self.model(sample)
                loss = self.loss_fn(logits, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.verbose:
                print(f'Epoch {e+1}, Loss: {loss.item():.4f}')

    def evaluate(self, df: pd.DataFrame, target: pd.Series) -> dict:
        """Evaluate the metrics: accuracy.
        
        Args:
            df (pandas.DataFrame): A dataframe with sensor data. Index has 
                two columns: `run_id` and `sample`. All other columns a value of 
                sensors.
            target (pandas.Series): A series with target values. Indes has two
                columns: `run_id` and `sample`.
        
        Returns:
            dict: A dictionary with metrics where keys are names of metrics and
                values are values of metrics.
        """
        dataset = SlidingWindowDataset(df, target, window_size=self.window_size)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        target, pred = [], []
        for sample, _target in tqdm(
            self.dataloader, desc='Steps ...', leave=False, disable=(not self.verbose)
        ):
            sample = sample.to(self.device)
            target.append(_target)
            with torch.no_grad():
                pred.append(self.predict(sample))
        target = torch.concat(target).numpy()
        pred = torch.concat(pred).numpy()
        metrics = {
            'accuracy': accuracy(pred, target),
            'correct_daignosis_rate': correct_daignosis_rate(pred, target), 
            'true_positive_rate': true_positive_rate(pred, target),
            'false_positive_rate': false_positive_rate(pred, target),
        }
        self._store_atrifacts_inference(metrics)
        return metrics

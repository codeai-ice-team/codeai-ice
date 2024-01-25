from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tqdm.auto import trange, tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from ice.anomaly_detection.utils import SlidingWindowDataset
from ice.base import BaseModel
from ice.anomaly_detection.metrics import accuracy


class BaseAnomalyDetection(BaseModel, ABC):
    """Base class for all anomaly detection models."""

    @abstractmethod
    def __init__(
            self,
            window_size: int,
            batch_size: int,
            lr: float,
            num_epochs: int,
            device: str,
            verbose: bool,
            name: str,
            threshold: float = 0.95
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
        self._cfg.path_set(["TASK"], "anomaly_detection")

        self.window_size = window_size
        self.loss_fn = None
        self.preprocessing = False
        self.scaler = None
        self.threshold = None

    _param_conf_map = dict(BaseModel._param_conf_map,
            **{
                "window_size": ["MODEL", "WINDOW_SIZE"],
                "threshold": ["MODEL", "THRESHOLD"]
            }
        )

    def fit(self, df: pd.DataFrame):
        """ Method fit for training anomaly detection models.

        Args:
            df (pd.DataFrame): data without anomaly states
        """
        assert len(df) >= self.window_size, "window size is larger than the length of df."
        if self.preprocessing:
            self.scaler.fit(df)
            df.loc[:] = self.scaler.transform(df)
        self._create_model(df)
        self._train_nn(df)

    def _fit(self, df: pd.DataFrame):
        pass

    def _predict(self, sample: torch.Tensor) -> torch.Tensor:
        input = sample.to(self.device)
        output = self.model(input)
        return output.cpu()

    def _train_nn(self, df: pd.DataFrame):
        self.model.train()
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        dataset = SlidingWindowDataset(df, window_size=self.window_size)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        errors = []
        for e in trange(self.num_epochs, desc='Epochs ...', disable=(not self.verbose)):
            for sample in tqdm(self.dataloader, desc='Steps ...', leave=False, disable=(not self.verbose)):
                input = sample.to(self.device)
                output = self.model(input)
                loss = self.loss_fn(output, input)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                error = torch.sum(torch.abs(input - output), dim=(1, 2))
                errors = np.append(errors, error.detach().numpy())
            if self.verbose:
                print(f'Epoch {e+1}, Loss: {loss.item():.4f}')
        self.threshold_value = np.quantile(errors, self.threshold)

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
        assert df.shape[0] == target.shape[0], f"target is incompatible with df by the length: {df.shape[0]} and {target.shape[0]}."
        assert np.all(df.index == target.index), "target's index and df's index are not the same."
        assert df.index.names == (['run_id', 'sample']), "An index should contain columns `run_id` and `sample`."

        if self.preprocessing:
            df.loc[:] = self.scaler.transform(df)
        dataset = SlidingWindowDataset(df, window_size=self.window_size, target=target)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        target, pred = [], []
        for sample, _target in tqdm(
            self.dataloader, desc='Steps ...', leave=False, disable=(not self.verbose)
        ):
            input = sample.to(self.device)
            target.append(_target)
            with torch.no_grad():
                output = self.predict(input)
                error = torch.sum(torch.abs(input - output), dim=(1, 2))
                pred.append((error > self.threshold_value).float())
        target = torch.concat(target).numpy()
        pred = torch.concat(pred).numpy()
        metrics = {
            'accuracy': accuracy(pred, target)
        }
        self._store_atrifacts_inference(metrics)
        return metrics

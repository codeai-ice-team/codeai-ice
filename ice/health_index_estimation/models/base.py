from ice.base import BaseModel
from abc import ABC
import pandas as pd
from torch.optim import AdamW
import torch
from torch import nn

from ice.health_index_estimation.metrics import mse, rmse


class BaseHealthIndexEstimation (BaseModel, ABC):
    """Base class for all HI diagnosis models."""

    def _prepare_for_training(self, input_dim: int, output_dim: int):
        self.loss_fn = nn.L1Loss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

    def _predict(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.to(self.device)
        predicted_rul = self.model(sample)
        return predicted_rul.cpu()

    def _calculate_metrics(self, pred: torch.tensor, target: torch.tensor) -> dict:
        metrics = {
            "mse": mse(pred, target),
            "rmse": rmse(pred, target),
        }
        return metrics
    
    def _set_dims(self, df: pd.DataFrame, target: pd.Series):
        self.input_dim = df.shape[1]
        self.output_dim = 1

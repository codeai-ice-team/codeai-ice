from abc import ABC
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from ice.base import BaseModel
from ice.fault_diagnosis.metrics import (
    accuracy, correct_daignosis_rate, true_positive_rate, false_positive_rate)


class BaseFaultDiagnosis(BaseModel, ABC):
    """Base class for all fault diagnosis models."""

    def _prepare_for_training(self, input_dim: int, output_dim: int):
        weight = torch.ones(output_dim, device=self.device) * 0.5
        weight[1:] /= output_dim - 1
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def _predict(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.to(self.device)
        logits = self.model(sample)
        return logits.argmax(axis=1).cpu()

    def _calculate_metrics(self, pred: torch.tensor, target: torch.tensor) -> dict:
        metrics = {
            'accuracy': accuracy(pred, target),
            'correct_daignosis_rate': correct_daignosis_rate(pred, target), 
            'true_positive_rate': true_positive_rate(pred, target),
            'false_positive_rate': false_positive_rate(pred, target),
        }
        return metrics
    
    def _set_dims(self, df: pd.DataFrame, target: pd.Series):
        self.input_dim = df.shape[1]
        self.output_dim = len(set(target))

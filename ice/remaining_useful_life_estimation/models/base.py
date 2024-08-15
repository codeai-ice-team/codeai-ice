from ice.base import BaseModel
from abc import ABC
import pandas as pd
from torch.optim import AdamW
import torch
from torch import nn

from ice.remaining_useful_life_estimation.metrics import rmse, cmapss_score

class RegLoss:
    def __init__(self, z_dims=2):
        self.name = "RegLoss"
        self.z_dims = z_dims
        self.criterion = nn.MSELoss()

    def __call__(self, y, y_hat):
        return self.criterion(y, y_hat)

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


class TotalLoss:
    def __init__(self):
        self.RegLoss = RegLoss()

    
    def __call__(self, model_output, y):

        if type(model_output) != torch.Tensor:
            
            y = y.to(torch.float64)
            
            loss = 2*self.RegLoss(y, model_output[0].to(torch.float64))
            for y_hat in model_output[1:]:
                loss += self.RegLoss(y, y_hat.to(torch.float64))

            return loss
        
        else:
            return self.RegLoss(y.to(torch.float64), model_output.to(torch.float64))



class BaseRemainingUsefulLifeEstimation(BaseModel, ABC):
    """Base class for all RUL models."""

    def _prepare_for_training(self, input_dim: int, output_dim: int):
        self.loss_fn = TotalLoss()# nn.L1Loss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98))

    def _predict(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.to(self.device)
        predicted_rul = self.model(sample)
        return predicted_rul.cpu()

    def _calculate_metrics(self, pred: torch.tensor, target: torch.tensor) -> dict:
        metrics = {
            "rmse": rmse(pred, target),
            "cmapss_score": cmapss_score(pred, target),
        }
        return metrics
    
    def _set_dims(self, df: pd.DataFrame, target: pd.Series):
        self.input_dim = df.shape[1]
        self.output_dim = 1

import numpy as np
import pandas as pd
import pytest
from ice.base import BaseDataset
from ice.anomaly_detection import models as ad_models
from ice.anomaly_detection import datasets as ad_datasets
from ice.anomaly_detection.metrics import (
    accuracy)
from inspect import getmembers, isclass
import torch

models = [f[1] for f in getmembers(ad_models, isclass)]
datasets = [f[1] for f in getmembers(ad_datasets, isclass) 
    if issubclass(f[1], BaseDataset) and f[1] != BaseDataset]


class TestOnSyntheticData:
    def setup_class(self):
        df0 = pd.DataFrame({
            'sensor_0': np.sin(np.linspace(0, 20, 100)),
            'sensor_1': np.sin(np.linspace(0, 10, 100)),
            'sample': np.arange(100),
            'run_id': 0,
        })
        df1 = pd.DataFrame({
            'sensor_0': np.sin(np.linspace(0, 30, 100)),
            'sensor_1': np.sin(np.linspace(0, 15, 100)),
            'sample': np.arange(100),
            'run_id': 1,
        })
        self.window_size = 10
        self.num_sensors = 2
        self.df = pd.concat([df0, df1]).set_index(['run_id', 'sample'])
        self.target = pd.Series([0] * 100 + [1] * 100, index=self.df.index)

    @pytest.mark.parametrize("model_class", models)
    def test_exception(self, model_class):
        self.model = model_class(window_size=self.window_size)
        self.model._create_model(self.df)
        self.model.fit(self.df[:100])

        with pytest.raises(Exception) as exc_info:
            self.model.evaluate(self.df.iloc[:10], self.target.iloc[:11])
        assert "target is incompatible with df by the length" in str(exc_info.value)
        with pytest.raises(Exception) as exc_info:
            _df = self.df.reset_index()
            self.model.evaluate(_df, self.target)
        assert "target's index and df's index are not the same." in str(exc_info.value)
        with pytest.raises(Exception) as exc_info:
            _df = self.df.copy()
            _df.index.names = (['index0', 'index1'])
            _target = self.target.copy()
            _target.index.names = (['index0', 'index1'])
            self.model.evaluate(_df, _target)
        assert "An index should contain columns `run_id` and `sample`." in str(exc_info.value)

    @pytest.mark.parametrize("model_class", models)
    def test_fit(self, model_class):
        self.model = model_class(window_size=self.window_size)
        self.model._create_model(self.df)
        self.model.fit(self.df[:100])
        assert True

    @pytest.mark.parametrize("model_class", models)
    def test_eval(self, model_class):
        self.model = model_class(window_size=self.window_size)
        self.model._create_model(self.df)
        self.model.fit(self.df[:100])
        metrics = self.model.evaluate(self.df, self.target)
        assert metrics['accuracy'] >= 0

    @pytest.mark.parametrize("model_class", models)
    def test_predict(self, model_class):
        self.model = model_class(window_size=self.window_size)
        self.model._create_model(self.df)
        sample = torch.randn(16, self.window_size, self.num_sensors)
        pred_target = self.model.predict(sample)
        assert pred_target.shape == sample.shape

    def test_metrics(self):
        np.random.seed(0)
        pred = np.random.permutation(self.target)
        assert round(accuracy(pred, self.target), 4) == 0.47


@pytest.mark.parametrize("dataset_class", datasets)
def test_dataset_loading(dataset_class):
    with pytest.raises(Exception) as exc_info:
        dataset_class(num_chunks=1, force_download=True)
    assert "File is not a zip file" in str(exc_info.value)


def test_dataset_small_tep():
    ad_datasets.AnomalyDetectionSmallTEP(force_download=True)
    assert True
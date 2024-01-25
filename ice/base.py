from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import os
import zipfile
import requests
import datetime
import json
from ice.configs import Config


class BaseDataset(ABC):
    """Base class for datasets."""
    def __init__(self, num_chunks=None, force_download=False):
        """
        Args:
            num_chunks (int): If given, download only num_chunks chunks of data.
                Used for testing purposes.
            force_download (bool): If True, download the dataset even if it exists.
        """
        self.df = None
        self.target = None
        self.train_mask = None
        self.test_mask = None
        self.name = None
        self.public_link = None
        self.set_name_public_link()
        self._load(num_chunks, force_download)
    
    def set_name_public_link(self):
        """
        This method has to be implemented by all children. Set name and public link. 
        """
        pass
    
    def _load(self, num_chunks, force_download):
        """Load the dataset in self.df and self.target."""
        ref_path = f'data/{self.name}/'
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
        zfile_path = f'data/{self.name}.zip'

        url = self._get_url(self.public_link)
        if not os.path.exists(zfile_path) or force_download:
            self._download_pgbar(url, zfile_path, self.name, num_chunks)
            
        self._extracting_files(zfile_path, ref_path)
        self.df = self._read_csv_pgbar(ref_path + 'df.csv', index_col=['run_id', 'sample'])
        self.target = self._read_csv_pgbar(ref_path + 'target.csv', index_col=['run_id', 'sample'])['target']
        self.train_mask = self._read_csv_pgbar(ref_path + 'train_mask.csv', index_col=['run_id', 'sample'])['train_mask']
        self.train_mask = self.train_mask.astype(bool)
        self.test_mask = ~self.train_mask
    
    def _get_url(self, public_link):
        r = requests.get(f'https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_link}')
        return r.json()['file']

    def _read_csv_pgbar(self, csv_path, index_col, chunk_size=1024*100):
        rows = sum(1 for _ in open(csv_path, 'r')) - 1
        chunk_list = []
        with tqdm(total=rows, desc=f'Reading {csv_path}') as pbar:
            for chunk in pd.read_csv(csv_path, index_col=index_col, chunksize=chunk_size):
                chunk_list.append(chunk)
                pbar.update(len(chunk))
        df = pd.concat((f for f in chunk_list), axis=0)
        return df

    def _download_pgbar(self, url, zfile_path, fname, num_chunks):
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("Content-Length"))
        with open(zfile_path, 'wb') as file: 
            with tqdm(
                total=total,
                desc=f'Downloading {fname}',
                unit='B',
                unit_scale=True,
                unit_divisor=1024) as pbar:
                i = 0    
                for data in resp.iter_content(chunk_size=1024):
                    if num_chunks is not None and num_chunks == i:
                        break
                    file.write(data)
                    pbar.update(len(data))
                    i += 1

    def _extracting_files(self, zfile_path, ref_path, block_size=1024*10000):
        with zipfile.ZipFile(zfile_path, 'r') as zfile:
            for entry_info in zfile.infolist():
                if os.path.exists(ref_path + entry_info.filename):
                    continue
                input_file = zfile.open(entry_info.filename)
                target_file = open(ref_path + entry_info.filename, 'wb')
                block = input_file.read(block_size)
                with tqdm(
                    total=entry_info.file_size, 
                    desc=f'Extracting {entry_info.filename}', 
                    unit='B', 
                    unit_scale=True, 
                    unit_divisor=1024) as pbar:
                    while block:
                        target_file.write(block)
                        block = input_file.read(block_size)
                        pbar.update(block_size)
                input_file.close()
                target_file.close()


class BaseModel(ABC):
    """Base class for all models."""

    _param_conf_map = {
            "batch_size" : ["DATASET", "BATCH_SIZE"],
            "lr" : ["OPTIMIZATION", "LR"],
            "num_epochs" : ["OPTIMIZATION", "NUM_EPOCHS"],
            "verbose" : ["VERBOSE"],
            "device" : ["DEVICE"],
            "name" : ["EXPERIMENT_NAME"]
        }

    @abstractmethod
    def __init__(
            self, 
            batch_size: int, 
            lr: float, 
            num_epochs: int, 
            device: str, 
            verbose: bool,
            name: str
        ):
        """
        Args:
            batch_size (int): The batch size to train the model.
            lr (float): The larning rate to train the model.
            num_epochs (float): The number of epochs to train the model.
            device (str): The name of a device to train the model. `cpu` and 
                `cuda` are possible.
            verbose (bool): If true, show the progress bar in training.
            name (str): The name of the model for artifact storing.
        """
        self._cfg = Config()
        self._cfg.path_set(["MODEL", "NAME"], self.__class__.__name__)
        self._output_dir = "outputs"

        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.verbose = verbose
        self.model = None
        self.name = name

    def __setattr__(self, __name: str, __value: Any):
        if __name not in ['cfg', 'param_conf_map'] and __name in self._param_conf_map.keys():
            self._cfg.path_set(self._param_conf_map[__name], __value)

        super().__setattr__(__name, __value)
        if __name == "name":
            self._training_path, self._inference_path = self._initialize_paths()
    
    @classmethod
    def from_config(cls, cfg : Config):
        """Create instance of the model class with parameters from config.

        Args:
            cfg (Config): A config with model's parameters.

        Returns:
            BaseModel: Instance of BaseModel child class initialized with parameters from config.
        """

        param_dict = {}

        for key in cls._param_conf_map.keys():
            param_dict[key] = cfg.path_get(cls._param_conf_map[key])
        
        return cls(**param_dict)

    def fit(self, df: pd.DataFrame, target: pd.Series):
        """Fit (train) the model by a given dataset.

        Args:
            df (pandas.DataFrame): A dataframe with sensor data. Index has 
                two columns: `run_id` and `sample`. All other columns a value of 
                sensors.
            target (pandas.Series): A series with target values. Indes has two
                columns: `run_id` and `sample`.
        """
        assert df.shape[0] == target.shape[0], f"target is incompatible with df by the length: {df.shape[0]} and {target.shape[0]}."
        assert np.all(df.index == target.index), "target's index and df's index are not the same."
        assert df.index.names == (['run_id', 'sample']), "An index should contain columns `run_id` and `sample`."
        self._create_model(df, target)
        assert self.model is not None, "Model creation error."
        self._fit(df, target)
        self._store_atrifacts_train()

    def _initialize_paths(self):
        artifacts_path = os.path.join(self._output_dir, self.name)
        training_path = os.path.join(artifacts_path, 'training')
        inference_path = os.path.join(artifacts_path, 'inference')

        return training_path, inference_path
    
    def _store_atrifacts_train(self):
        save_path = os.path.join(self._training_path, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(save_path, exist_ok= True)
        self._cfg.to_yaml(os.path.join(save_path, 'config.yaml'))

    def _store_atrifacts_inference(self, metrics):
        save_path = os.path.join(self._inference_path, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(save_path, exist_ok= True)

        self._cfg.to_yaml(os.path.join(save_path, 'config.yaml'))
        with open(os.path.join(save_path, "metrics.json"), 'w') as json_file:
            json.dump(metrics, json_file)

    @abstractmethod
    def _create_model(self, df: pd.DataFrame, target: pd.Series):
        """
        This method has to be implemented by all children. Create a torch 
        model for traing and prediction. 
        """
        pass

    @abstractmethod
    def _fit(self, df: pd.DataFrame, target: pd.Series):
        """
        This method has to be implemented by all children. Fit (train) the model 
        by a given dataset.
        """
        pass

    @torch.no_grad()
    def predict(self, sample: torch.Tensor) -> torch.Tensor:
        """Make a prediction for a given batch of samples.

        Args:
            sample (torch.Tensor): A tensor of the shape (B, L, C) where
                B is the batch size, L is the sequence length, C is the number
                of sensors.
        
        Returns:
            torch.Tensor: A tensor with predictions of the shape (B,).
        """
        self.model.eval()
        self.model.to(self.device)
        return self._predict(sample)

    @abstractmethod
    def _predict(self, sample: torch.Tensor) -> torch.Tensor:
        """
        This method has to be implemented by all children. Make a prediction 
        for a given batch of samples.
        """
        pass
    
    @abstractmethod
    def evaluate(self, df: pd.DataFrame, target: pd.Series) -> dict:
        """This method has to be implemented by all children. Evaluate the 
        metrics. Docstring has to be rewritten so that all metrics are clearly 
        described. 

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
        pass

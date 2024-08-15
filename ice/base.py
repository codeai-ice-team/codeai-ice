from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm, trange
import os
import zipfile
import requests
import datetime
import json
import random
from ice.configs import Config
import time
from torch.utils.data import DataLoader, Dataset, random_split
import optuna


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
        ref_path = f"data/{self.name}/"
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
        zfile_path = f"data/{self.name}.zip"

        url = self._get_url(self.public_link)
        if not os.path.exists(zfile_path) or force_download:
            self._download_pgbar(url, zfile_path, self.name, num_chunks)

        self._extracting_files(zfile_path, ref_path)
        self.df = self._read_csv_pgbar(
            ref_path + "df.csv", index_col=["run_id", "sample"]
        )
        self.target = self._read_csv_pgbar(
            ref_path + "target.csv", index_col=["run_id", "sample"]
        )["target"]
        self.train_mask = self._read_csv_pgbar(
            ref_path + "train_mask.csv", index_col=["run_id", "sample"]
        )["train_mask"]
        self.train_mask = self.train_mask.astype(bool)
        self.test_mask = ~self.train_mask

    def _get_url(self, public_link):
        url = ""
        r = requests.get(
            f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_link}"
        )
        if r.status_code == 200:
            url = r.json()["file"]
        else:
            raise Exception(r.json()["description"])
        return url

    def _read_csv_pgbar(self, csv_path, index_col, chunk_size=1024 * 100):
        rows = sum(1 for _ in open(csv_path, "r")) - 1
        chunk_list = []
        with tqdm(total=rows, desc=f"Reading {csv_path}") as pbar:
            for chunk in pd.read_csv(
                csv_path, index_col=index_col, chunksize=chunk_size
            ):
                chunk_list.append(chunk)
                pbar.update(len(chunk))
        df = pd.concat((f for f in chunk_list), axis=0)
        return df

    def _download_pgbar(self, url, zfile_path, fname, num_chunks):
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("Content-Length"))
        with open(zfile_path, "wb") as file:
            with tqdm(
                total=total,
                desc=f"Downloading {fname}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                i = 0
                for data in resp.iter_content(chunk_size=1024):
                    if num_chunks is not None and num_chunks == i:
                        break
                    file.write(data)
                    pbar.update(len(data))
                    i += 1

    def _extracting_files(self, zfile_path, ref_path, block_size=1024 * 10000):
        with zipfile.ZipFile(zfile_path, "r") as zfile:
            for entry_info in zfile.infolist():
                if os.path.exists(ref_path + entry_info.filename):
                    continue
                input_file = zfile.open(entry_info.filename)
                target_file = open(ref_path + entry_info.filename, "wb")
                block = input_file.read(block_size)
                with tqdm(
                    total=entry_info.file_size,
                    desc=f"Extracting {entry_info.filename}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    while block:
                        target_file.write(block)
                        block = input_file.read(block_size)
                        pbar.update(block_size)
                input_file.close()
                target_file.close()


class BaseModel(ABC):
    """Base class for all models."""

    _param_conf_map = {
        "batch_size": ["DATASET", "BATCH_SIZE"],
        "lr": ["OPTIMIZATION", "LR"],
        "num_epochs": ["OPTIMIZATION", "NUM_EPOCHS"],
        "verbose": ["VERBOSE"],
        "device": ["DEVICE"],
        "name": ["EXPERIMENT_NAME"],
        "window_size": ["MODEL", "WINDOW_SIZE"],
        "stride": ["MODEL", "STRIDE"],
        "val_ratio": ["DATASET", "VAL_RATIO"],
        "random_seed": ["SEED"],
    }

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
    ):
        """
        Args:
            window_size (int): The window size to train the model.
            stride (int): The time interval between first points of consecutive
                sliding windows in training.
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
        self._cfg = Config()
        self._cfg.path_set(["MODEL", "NAME"], self.__class__.__name__)
        self._output_dir = "outputs"

        self.window_size = window_size
        self.val_ratio = val_ratio
        self.val_metrics = None
        self.stride = stride
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device
        self.verbose = verbose
        self.model = None
        self.name = name
        self.random_seed = random_seed
        self.save_checkpoints = save_checkpoints
        self.input_dim = None
        self.output_dim = None
        self.train_time = "no date"
        self.checkpoint_epoch = 0
        self.direction = "minimize"

    def __setattr__(self, __name: str, __value: Any):
        if (
            __name not in ["cfg", "param_conf_map"]
            and __name in self._param_conf_map.keys()
        ):
            self._cfg.path_set(self._param_conf_map[__name], __value)

        super().__setattr__(__name, __value)
        if __name == "name":
            self._training_path, self._inference_path, self._checkpoints_path = (
                self._initialize_paths()
            )

    @classmethod
    def from_config(cls, cfg: Config):
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

    def fit(
        self,
        df: pd.DataFrame,
        target: pd.Series = None,
        epochs: int = None,
        save_path: str = None,
        trial: optuna.Trial = None,
        force_model_ctreation: bool = False,
    ):
        """Fit (train) the model by a given dataset.

        Args:
            df (pandas.DataFrame): A dataframe with sensor data. Index has
                two columns: `run_id` and `sample`. All other columns a value of
                sensors.
            target (pandas.Series): A series with target values. Index has two
                columns: `run_id` and `sample`. It is omitted for anomaly
                detection task.
            epochs (int): The number of epochs for training step. If None,
                self.num_epochs parameter is used.
            save_path (str): Path to save checkpoints. If None, the path is
                created automatically.
            trial (optuna.Trial, None): optuna.Trial object created by optimize method.
            force_model_ctreation (bool): force fit to create model for optimization study.
        """
        self.train_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if epochs is None:
            epochs = self.num_epochs
        self._validate_inputs(df, target)
        if self.model is None or force_model_ctreation:
            self._set_dims(df, target)
            self._create_model(self.input_dim, self.output_dim)
            assert self.model is not None, "Model creation error."
            self._prepare_for_training(self.input_dim, self.output_dim)
        self._train_nn(
            df=df, target=target, epochs=epochs, save_path=save_path, trial=trial
        )
        self._store_atrifacts_train()

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

    def optimize(
        self,
        df: pd.DataFrame,
        target: pd.Series = None,
        optimize_parameter: str = "batch_size",
        optimize_range: tuple = (128, 256),
        direction: str = "minimize",
        n_trials: int = 5,
        epochs: int = None,
        optimize_metric: str = None,
    ):
        """Make the optuna study to return the best hyperparameter value on validation dataset

        Args:
            df (pd.DataFrame): DataFrame to use method fit
            optimize_parameter (str, optional): Model parameter to optimize. Defaults to 'batch_size'.
            optimize_range (tuple, optional): Model parameter range for optuna trials. Defaults to (128, 256).
            n_trials (int, optional): number of trials. Defaults to 5.
            target (pd.Series, optional): target pd.Series to use method fit. Defaults to None.
            epochs (int, optional): Epoch number to use method fit. Defaults to None.
            optimize_metric (str): Metric on validation dataset to use as a target for hyperparameter optimization.
            direction (str): "minimize" or "maximize" the target for hyperparameter optimization

        """
        param_type = type(self.__dict__[optimize_parameter])
        self.direction = direction

        defaults_torch_backends = (
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        )
        self.dump = self._training_path

        # make torch deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        if not epochs:
            epochs = self.num_epochs

        def objective(trial):

            self._training_path = self.dump
            """optuna objective

            Args:
                trial (optuna.Trial): optuna trial object

            Raises:
                AssertionError: Returns if optimize_parameter value is not numerical

            Returns:
                float: best validation loss to perform optimization step
            """
            if param_type == float:
                suggested_param = trial.suggest_float(
                    optimize_parameter, optimize_range[0], optimize_range[1]
                )
            elif param_type == int:
                suggested_param = trial.suggest_int(
                    optimize_parameter, optimize_range[0], optimize_range[1]
                )
            elif optimize_parameter == "lr":
                suggested_param = trial.suggest_loguniform(
                    optimize_parameter, optimize_range[0], optimize_range[1]
                )
            else:
                raise AssertionError(f"{optimize_parameter} is not int or float value")

            setattr(self, optimize_parameter, suggested_param)

            if optimize_metric:
                self.val_metrics = True

            self._training_path = (
                self._training_path + f"/parameter_{optimize_parameter} optimization"
            )
            os.makedirs(self._training_path, exist_ok=True)

            self.checkpoint_epoch = 0
            print(f"trial step with {optimize_parameter} = {suggested_param}")
            self.fit(
                df=df,
                epochs=epochs,
                target=target,
                trial=trial,
                force_model_ctreation=True,
            )
            # use the best key metric on validation dataset from training
            # if there is no such metric, use the best validation loss

            if optimize_metric:
                return self.best_validation_metrics[
                    optimize_metric
                ]  # dict with all best metric achieved during training -> write it
            else:
                return self.best_val_loss

        study = optuna.create_study(
            direction=direction,
            pruner=optuna.pruners.PercentilePruner(25.0),
            study_name=f"/parameter_{optimize_parameter} study",
        )
        study.optimize(objective, n_trials=n_trials)
        self._training_path = self.dump

        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best trial: {study.best_trial}")

        df_trials = study.trials_dataframe()
        df_trials.to_csv(
            self._training_path
            + f"/parameter_{optimize_parameter} optimization"
            + f"/parameter_{optimize_parameter}.csv",
            index=False,
        )

        # restore standard torch deterministic values
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = (
            defaults_torch_backends
        )
        del os.environ["CUBLAS_WORKSPACE_CONFIG"]
        torch.use_deterministic_algorithms(False)

    @torch.no_grad()
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
        self._validate_inputs(df, target)
        dataset = SlidingWindowDataset(df, target, window_size=self.window_size)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        target, pred = [], []
        for sample, _target in tqdm(
            self.dataloader, desc="Steps ...", leave=False, disable=(not self.verbose)
        ):
            sample = sample.to(self.device)
            target.append(_target)
            pred.append(self.predict(sample))
        target = torch.concat(target).numpy()
        pred = torch.concat(pred).numpy()
        metrics = self._calculate_metrics(pred, target)
        self._store_atrifacts_inference(metrics)
        return metrics

    @torch.no_grad()
    def model_param_estimation(self):
        """Calculate number of self.model parameters, mean and std for inference time

        Returns:
            tuple: A tuple containing the number of parameters in the
                   model and the mean and standard deviation of model inference time.
        """
        assert (
            self.model != None
        ), "use model.fit() to create fitted model object before"
        sample = iter(self.dataloader).__next__()
        if len(sample) == 2:
            x, y = sample
        else:
            x = sample

        dummy_input = torch.randn(1, x.shape[1], x.shape[2]).to(self.device)
        repetitions = 500
        times = np.zeros((repetitions, 1))

        if self.device == "cuda":
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )

            for i in range(repetitions):
                starter.record()
                _ = self.model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                times[i] = curr_time

        else:
            for i in range(repetitions):
                start_time = time.time()
                _ = self.model(dummy_input)
                end_time = time.time()

                curr_time = (end_time - start_time) * 1000  # Convert to milliseconds
                times[i] = curr_time

        mean_inference_time = np.sum(times) / repetitions
        std_inference_time = np.std(times)

        num_params = sum(p.numel() for p in self.model.parameters())

        return num_params, (mean_inference_time, std_inference_time)

    @abstractmethod
    def _create_model(self, input_dim: int, output_dim: int):
        """
        This method has to be implemented by all children. Create a torch
        model for traing and prediction.
        """
        pass

    @abstractmethod
    def _prepare_for_training(self, input_dim: int, output_dim: int):
        """
        This method has to be implemented by all children. Prepare the model
        for training by a given dataset.
        """
        pass

    @abstractmethod
    def _calculate_metrics(self, pred: torch.tensor, target: torch.tensor) -> dict:
        """
        This method has to be implemented by all children. Calculate metrics.
        """
        pass

    @abstractmethod
    def _predict(self, sample: torch.Tensor) -> torch.Tensor:
        """
        This method has to be implemented by all children. Make a prediction
        for a given batch of samples.
        """
        pass

    @abstractmethod
    def _set_dims(self, df: pd.DataFrame, target: pd.Series):
        """
        This method has to be implemented by all children. Calculate input and
        output dimensions of the model by the given dataset.
        """
        pass

    def _validate_inputs(self, df: pd.DataFrame, target: pd.Series):
        assert (
            df.shape[0] == target.shape[0]
        ), f"target is incompatible with df by the length: {df.shape[0]} and {target.shape[0]}."
        assert np.all(
            df.index == target.index
        ), "target's index and df's index are not the same."
        assert df.index.names == (
            ["run_id", "sample"]
        ), "An index should contain columns `run_id` and `sample`."
        assert (
            len(df) >= self.window_size
        ), "window size is larger than the length of df."
        assert len(df) >= self.stride, "stride is larger than the length of df."

    def _train_nn(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        epochs: int,
        save_path: str,
        trial: optuna.Trial,
    ):
        self.model.train()
        self.model.to(self.device)
        self.best_val_loss = (
            float("inf") if self.direction == "minimize" else float("-inf")
        )
        self.best_validation_metrics = {}
        self._set_seed()

        dataset = SlidingWindowDataset(df, target, window_size=self.window_size, stride=self.stride)
        val_size = max(int(len(dataset) * self.val_ratio), 1)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        self.dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        for e in trange(
            self.checkpoint_epoch,
            self.checkpoint_epoch + epochs,
            desc="Epochs ...",
            disable=(not self.verbose),
        ):
            for sample, target in tqdm(
                self.dataloader,
                desc="Steps ...",
                leave=False,
                disable=(not self.verbose),
            ):
                sample = sample.to(self.device)
                target = target.to(self.device)
                logits = self.model(sample)
                loss = self.loss_fn(logits, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.verbose:
                print(f"Epoch {e+1}, Loss: {loss.item():.4f}")
            self.checkpoint_epoch = e + 1
            if self.save_checkpoints:
                self.save_checkpoint(save_path)

            val_loss, val_metrics = self._validate_nn()

            if self.direction == "minimize":
                self.best_val_loss = (
                    val_loss if val_loss < self.best_val_loss else self.best_val_loss
                )
            else:
                self.best_val_loss = (
                    val_loss if val_loss > self.best_val_loss else self.best_val_loss
                )

            if self.val_metrics:
                for key, value in val_metrics.items():
                    if key not in self.best_validation_metrics:
                        self.best_validation_metrics[key] = value
                    else:
                        if self.direction == "minimize":
                            self.best_validation_metrics[key] = (
                                value
                                if self.best_validation_metrics[key] > value
                                else self.best_validation_metrics[key]
                            )
                        else:
                            self.best_validation_metrics[key] = (
                                value
                                if self.best_validation_metrics[key] < value
                                else self.best_validation_metrics[key]
                            )

            if self.verbose:
                if self.val_metrics:
                    print(
                        f"Epoch {e+1}, Validation Loss: {val_loss:.4f}, Metrics: {val_metrics}"
                    )
                else:
                    print(f"Epoch {e+1}, Validation Loss: {val_loss:.4f}")

            if trial:
                trial.report(val_loss, self.checkpoint_epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    def _validate_nn(self):
        self.model.eval()

        val_loss = 0
        target_list, pred_list = [], []
        with torch.no_grad():
            for sample, target in self.val_dataloader:
                sample = sample.to(self.device)
                target = target.to(self.device)
                logits = self.model(sample)
                loss = self.loss_fn(logits, target)
                val_loss += loss.item()

                if self.val_metrics:
                    pred = self.predict(sample)
                    target_list.append(target.cpu())
                    pred_list.append(pred.cpu())

        val_loss /= len(self.val_dataloader)

        if self.val_metrics:
            target_tensor = torch.cat(target_list).numpy()
            pred_tensor = torch.cat(pred_list).numpy()

            val_metrics = self._calculate_metrics(pred_tensor, target_tensor)
        else:
            val_metrics = None

        self.model.train()

        return val_loss, val_metrics

    def _set_seed(self):
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def _initialize_paths(self):
        artifacts_path = os.path.join(self._output_dir, self.name)
        training_path = os.path.join(artifacts_path, "training")
        inference_path = os.path.join(artifacts_path, "inference")
        checkpoints_path = os.path.join(artifacts_path, "checkpoints")
        return training_path, inference_path, checkpoints_path

    def _store_atrifacts_train(self):
        save_path = os.path.join(self._training_path, self.train_time)
        os.makedirs(save_path, exist_ok=True)
        self._cfg.to_yaml(os.path.join(save_path, "config.yaml"))

    def _store_atrifacts_inference(self, metrics):
        save_path = os.path.join(self._inference_path, self.train_time)
        os.makedirs(save_path, exist_ok=True)

        self._cfg.to_yaml(os.path.join(save_path, "config.yaml"))
        with open(os.path.join(save_path, "metrics.json"), "w") as json_file:
            json.dump(metrics, json_file)

    def save_checkpoint(self, save_path: str = None):
        """Save checkpoint.

        Args:
            save_path (str): Path to save checkpoint.
        """
        if save_path is None:
            checkpoints_path = os.path.join(self._checkpoints_path, self.train_time)
            os.makedirs(checkpoints_path, exist_ok=True)
            file_path = self.name + "_epoch_" + str(self.checkpoint_epoch)
            save_path = os.path.join(checkpoints_path, file_path + ".tar")
        torch.save(
            {
                "config": self._cfg,
                "epoch": self.checkpoint_epoch,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Path to load checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._cfg = checkpoint["config"]
        self.input_dim = checkpoint["input_dim"]
        self.output_dim = checkpoint["output_dim"]
        self._create_model(self.input_dim, self.output_dim)
        self.model.to(self.device)
        assert self.model is not None, "Model creation error."
        self._prepare_for_training(self.input_dim, self.output_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.checkpoint_epoch = checkpoint["epoch"]


class SlidingWindowDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, target: pd.Series, window_size: int, stride: int = 1
    ):
        self.df = df
        self.target = target
        self.window_size = window_size

        window_end_indices = []
        run_ids = df.index.get_level_values(0).unique()
        for run_id in tqdm(run_ids, desc="Creating sequence of samples"):
            indices = np.array(df.index.get_locs([run_id]))
            indices = indices[self.window_size :: stride]
            window_end_indices.extend(indices)
        self.window_end_indices = np.array(window_end_indices)

    def __len__(self):
        return len(self.window_end_indices)

    def __getitem__(self, idx):
        window_index = self.window_end_indices[idx]
        sample = self.df.values[window_index - self.window_size : window_index]
        if self.target is not None:
            target = self.target.values[window_index]
        else:
            target = sample.astype("float32")
        return sample.astype("float32"), target

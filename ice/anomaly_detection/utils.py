from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class SlidingWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int, target: pd.Series = None):
        self.df = df
        self.target = target
        self.window_size = window_size

        window_end_indices = []
        run_ids = df.index.get_level_values(0).unique()
        for run_id in tqdm(run_ids, desc='Creating sequence of samples'):
            indices = np.array(df.index.get_locs([run_id]))
            indices = indices[self.window_size:]
            window_end_indices.extend(indices)
        self.window_end_indices = np.array(window_end_indices)

    def __len__(self):
        return len(self.window_end_indices)

    def __getitem__(self, idx):
        window_index = self.window_end_indices[idx]
        sample = self.df.values[window_index - self.window_size:window_index]
        if self.target is not None:
            target = self.target.values[window_index]
            return sample.astype(np.float32), target
        return sample.astype(np.float32)

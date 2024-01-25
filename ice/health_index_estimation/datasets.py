from ice.base import BaseDataset
import os
import pandas as pd

from copy import copy
from scipy import interpolate


class Milling(BaseDataset):
    """
    Preprocessed to mil data from the Milling Data Set:
    https://data.nasa.gov/Raw-Data/Milling-Wear/vjv9-9f3x/data

    """

    def set_name_public_link(self):
        self.name = "milling"
        self.public_link = "https://disk.yandex.ru/d/jnYLUicx6TIkVw"

    def _load(self, num_chunks, force_download):
        """
        Load the dataset in list obects: self.df, self.target, self.test and self.test_target.
        16 subdatasets, list index corresponds to a subdataset number in test or train part

        Benchmark preparation with fixed train-test and the paper-based interpolation
        of missing values

        """
        ref_path = f"data/{self.name}/"
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
        zfile_path = f"data/{self.name}.zip"

        url = self._get_url(self.public_link)
        if not os.path.exists(zfile_path) or force_download:
            self._download_pgbar(url, zfile_path, self.name, num_chunks)

        self._extracting_files(zfile_path, ref_path)

        # test and train subset number of cuts
        train_nums = [1, 3, 5, 7, 8, 9, 10, 11, 12, 13]
        test_nums = [2, 4, 6]

        data = [
            self._read_csv_pgbar(
                ref_path + f"case_{i+1}.csv", index_col=["cut_no", "sample"]
            )
            for i in range(16)
        ]

        inter_func = []

        for i in range(15):
            y = data[i].dropna().VB
            x = data[i].dropna().time

            f = interpolate.interp1d(x, y, assume_sorted=True, fill_value="extrapolate")

            data[i].VB = f(data[i].time)
            if i == 6:
                midpoint = len(data[i]) // 2
                data[i]["VB"].iloc[midpoint:].fillna(0.52, inplace=True)
                data[i]["VB"].iloc[:midpoint].fillna(0, inplace=True)
            else:
                data[i] = data[i].fillna(0)

        self.df = [data[i].drop(columns=["VB"]) for i in train_nums]
        self.target = [data[i]["VB"] for i in train_nums]

        self.test = [data[i].drop(columns=["VB"]) for i in test_nums]
        self.test_target = [data[i]["VB"] for i in test_nums]

    def _read_csv_pgbar(self, csv_path, index_col, chunksize=1024 * 100):
        df = pd.read_csv(csv_path)
        df.rename(columns={"cut_no": "run_id"}, inplace=True)
        df = df.set_index(["run_id", "sample"]).drop(
            columns=["Unnamed: 0", "case", "run"]
        )
        df["material"] = df["material"].astype("float64")

        return df

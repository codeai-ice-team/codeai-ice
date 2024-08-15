from ice.base import BaseDataset
import os
import pandas as pd


class RulCmapss(BaseDataset):
    """
    Preprocessed to piece wise RUL data from the dataset:
    Saxena A. et al. Damage propagation modeling for aircraft engine run-to-failure simulation
    DOI: 10.1109/PHM.2008.4711414

    """

    def set_name_public_link(self):
        self.name = "C-MAPSS"
        self.public_link = "https://disk.yandex.ru/d/payoj43vWTgLLw"

    def _load(self, num_chunks, force_download):
        """
        Load the dataset in list obects: self.df, self.target, self.test and self.test_target.
        4 subdatasets fd001-fd004, list index corresponds to a subdataset number

        """
        ref_path = f"data/{self.name}/"
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
        zfile_path = f"data/{self.name}.zip"

        url = self._get_url(self.public_link)
        if not os.path.exists(zfile_path) or force_download:
            self._download_pgbar(url, zfile_path, self.name, num_chunks)

        self._extracting_files(zfile_path, "data/")
        self.df = [
            self._read_csv_pgbar(
                ref_path + f"fd{i}_train.csv", index_col=["run_id", "sample"]
            ).drop(columns=["rul"])
            for i in range(1, 5)
        ]
        self.target = [
            self._read_csv_pgbar(
                ref_path + f"fd{i}_train.csv", index_col=["run_id", "sample"]
            )["rul"]
            for i in range(1, 5)
        ]
        self.test = [
            self._read_csv_pgbar(
                ref_path + f"fd{i}_test.csv", index_col=["run_id", "sample"]
            ).drop(columns=["rul"])
            for i in range(1, 5)
        ]
        self.test_target = [
            self._read_csv_pgbar(
                ref_path + f"fd{i}_test.csv", index_col=["run_id", "sample"]
            )["rul"]
            for i in range(1, 5)
        ]


class RulCmapssPaper(BaseDataset):
    """
    Preprocessed to piece wise RUL data from the dataset:
    Saxena A. et al. Damage propagation modeling for aircraft engine run-to-failure simulation
    DOI: 10.1109/PHM.2008.4711414. Target is the minimum rul value for every test device. 

    """

    def set_name_public_link(self):
        self.name = "C-MAPSS_paper_test"
        self.public_link = "https://disk.yandex.ru/d/IoUNSJMZQhVkpw"

    def _load(self, num_chunks, force_download):
        """
        Load the test dataset in list obects: self.df, self.target, self.test and self.test_target.
        4 subdatasets fd001-fd004, list index corresponds to a subdataset number

        """
        ref_path = f"data/{self.name}/"
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
        zfile_path = f"data/{self.name}.zip"

        url = self._get_url(self.public_link)
        if not os.path.exists(zfile_path) or force_download:
            self._download_pgbar(url, zfile_path, self.name, num_chunks)

        self._extracting_files(zfile_path, f"data/{self.name}/")
        
        self.test = [
            self._read_csv_pgbar(
                ref_path + f"fd{i}_test.csv", index_col=["run_id", "sample"]
            ).drop(columns=["rul"])
            for i in range(1, 5)
        ]
        self.test_target = [
            self._read_csv_pgbar(
                ref_path + f"/fd{i}_test.csv", index_col=["run_id", "sample"]
            )["rul"]
            for i in range(1, 5)
        ]

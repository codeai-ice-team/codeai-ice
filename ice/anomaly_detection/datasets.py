from ice.base import BaseDataset


class AnomalyDetectionSmallTEP(BaseDataset):
    """
    Cropped version of Tennessee Eastman Process dataset 
    Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017). 
    Additional Tennessee Eastman Process Simulation Data for 
    Anomaly Detection Evaluation (Version V1) [Computer software]. 
    Harvard Dataverse.
    https://doi.org/10.7910/DVN/6C3JR1.
    """
    def __init__(self, num_chunks=None, force_download=False):
        self.df = None
        self.test_mask = None
        self.name = None
        self.public_link = None
        self.set_name_public_link()
        self._load(num_chunks, force_download)
        self.target[self.target != 0] = 1
        self.train_mask[self.target != 0] = False

    def set_name_public_link(self):
        self.name = 'small_tep'
        self.public_link = 'https://disk.yandex.ru/d/DRiUmV2GyuTcjQ'


class AnomalyDetectionReinartzTEP(BaseDataset):
    """
    Dataset of Tennessee Eastman Process based on the paper Reinartz, C., 
    Kulahci, M., & Ravn, O. (2021). An extended Tennessee Eastman simulation 
    dataset for fault-detection and decision support systems. Computers & 
    Chemical Engineering, 149, 107281. 
    https://web.mit.edu/braatzgroup/links.html.
    """
    def __init__(self, num_chunks=None, force_download=False):
        self.df = None
        self.test_mask = None
        self.name = None
        self.public_link = None
        self.set_name_public_link()
        self._load(num_chunks, force_download)
        self.target[self.target != 0] = 1
        self.train_mask[self.target != 0] = False
        
    def set_name_public_link(self):
        self.name = 'reinartz_tep'
        self.public_link = 'https://disk.yandex.ru/d/NR6rjqCJCvrBZw'


class AnomalyDetectionRiethTEP(BaseDataset):
    """
    Dataset of Tennessee Eastman Process dataset 
    Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017). 
    Additional Tennessee Eastman Process Simulation Data for 
    Anomaly Detection Evaluation (Version V1) [Computer software]. 
    Harvard Dataverse. 
    https://doi.org/10.7910/DVN/6C3JR1.
    """
    def __init__(self, num_chunks=None, force_download=False):
        self.df = None
        self.test_mask = None
        self.name = None
        self.public_link = None
        self.set_name_public_link()
        self._load(num_chunks, force_download)
        self.target[self.target != 0] = 1
        self.train_mask[self.target != 0] = False

    def set_name_public_link(self):
        self.name = 'rieth_tep'
        self.public_link = 'https://disk.yandex.ru/d/l9C0HzQUw2Ying'

from ice.base import BaseDataset


class FaultDiagnosisRiethTEP(BaseDataset):
    """
    Dataset of Tennessee Eastman Process dataset 
    Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017). 
    Additional Tennessee Eastman Process Simulation Data for 
    Anomaly Detection Evaluation (Version V1) [Computer software]. 
    Harvard Dataverse. 
    https://doi.org/10.7910/DVN/6C3JR1.
    """
    def set_name_public_link(self):
        self.name = 'rieth_tep'
        self.public_link = 'https://disk.yandex.ru/d/l9C0HzQUw2Ying'


class FaultDiagnosisSmallTEP(BaseDataset):
    """
    Cropped version of Tennessee Eastman Process dataset 
    Rieth, C. A., Amsel, B. D., Tran, R., & Cook, M. B. (2017). 
    Additional Tennessee Eastman Process Simulation Data for 
    Anomaly Detection Evaluation (Version V1) [Computer software]. 
    Harvard Dataverse. 
    https://doi.org/10.7910/DVN/6C3JR1.
    """
    def set_name_public_link(self):
        self.name = 'small_tep'
        self.public_link = 'https://disk.yandex.ru/d/DRiUmV2GyuTcjQ'


class FaultDiagnosisReinartzTEP(BaseDataset):
    """
    Dataset of Tennessee Eastman Process based on the paper Reinartz, C., 
    Kulahci, M., & Ravn, O. (2021). An extended Tennessee Eastman simulation 
    dataset for fault-detection and decision support systems. Computers & 
    Chemical Engineering, 149, 107281. 
    https://web.mit.edu/braatzgroup/links.html.
    """
    def set_name_public_link(self):
        self.name = 'reinartz_tep'
        self.public_link = 'https://disk.yandex.ru/d/NR6rjqCJCvrBZw'

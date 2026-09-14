import numpy as np
import random
import torch

class Utils:
    def set_seed(self, seed):
        seed = int(seed)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def data_description(self, X, y):
        des_dict = {}
        des_dict['Samples'] = X.shape[0]
        des_dict['Features'] = X.shape[1]
        des_dict['Anomalies'] = sum(y)
        des_dict['Anomalies Ratio(%)'] = round((sum(y) / len(y)) * 100, 2)

        print(des_dict)

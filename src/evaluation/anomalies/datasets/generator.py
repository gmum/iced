import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.anomalies.utils import Utils


class DataGenerator:
    def __init__(self, seed:int=42, dataset:str=None, test_size:float=0.3,
                 generate_duplicates=True, n_samples_threshold=1000):
        '''
        :param seed: seed for reproducible results
        :param dataset: specific the dataset name
        :param test_size: testing set size
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_threshold will be dropped
        '''

        self.seed = seed
        self.dataset = dataset
        self.test_size = test_size

        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold

        # dataset list
        self.dataset_list_classical, self.dataset_list_cv, self.dataset_list_nlp = self.generate_dataset_list()

        # myutils function
        self.utils = Utils()

    def generate_dataset_list(self):
        def get_numeric_prefix(filename):
            base = os.path.splitext(filename)[0]
            prefix = base.split('_')[0]
            return int(prefix)

        # classical AD datasets
        dataset_list_classical = [
            os.path.splitext(_)[0] for _ in
            sorted(
                os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real/Classical')),
                key=get_numeric_prefix
            )
            if os.path.splitext(_)[1] == '.npz'
        ]

        # CV datasets
        dataset_list_cv = [os.path.splitext(_)[0] for _ in
                           os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real/CV_by_ResNet18'))
                           if os.path.splitext(_)[1] == '.npz']
        # NLP datasets
        dataset_list_nlp = [os.path.splitext(_)[0] for _ in
                            os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real/NLP_by_BERT'))
                            if os.path.splitext(_)[1] == '.npz']

        return dataset_list_classical, dataset_list_cv, dataset_list_nlp

    def generator(self, X=None, y=None, minmax=True, anomaly_stratify=False):
        # set seed for reproducible results
        self.utils.set_seed(self.seed)

        # load dataset
        if self.dataset is None:
            assert X is not None and y is not None, "For customized dataset, you should provide the X and y!"
            print('Testing on customized dataset...')
        else:
            if self.dataset in self.dataset_list_classical:
                dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real', 'Classical')
                data = np.load(os.path.join(dir_path, self.dataset + '.npz'), allow_pickle=True)
            elif self.dataset in self.dataset_list_cv:
                dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real', 'CV_by_ResNet18')
                data = np.load(os.path.join(dir_path, self.dataset + '.npz'), allow_pickle=True)
            elif self.dataset in self.dataset_list_nlp:
                dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real', 'NLP_by_BERT')
                data = np.load(os.path.join(dir_path, self.dataset + '.npz'), allow_pickle=True)
            else:
                raise NotImplementedError

            X = data['X']
            y = data['y']

        # if the dataset is too small, generating duplicate smaples up to n_samples_threshold
        if len(y) < self.n_samples_threshold and self.generate_duplicates:
            print(f'generating duplicate samples for dataset {self.dataset}...')
            self.utils.set_seed(self.seed)
            idx_duplicate = np.random.choice(np.arange(len(y)), self.n_samples_threshold, replace=True)
            X = X[idx_duplicate]
            y = y[idx_duplicate]

        # if the dataset is too large, subsampling for considering the computational cost
        if len(y) > 10000:
            print(f'subsampling for dataset {self.dataset}...')
            self.utils.set_seed(self.seed)
            anoms_len = 0
            while anoms_len < 2:
                idx_sample = np.random.choice(np.arange(len(y)), 10000, replace=False)
                X_temp = X[idx_sample]
                y_temp = y[idx_sample]
                anoms_len = len(np.where(y_temp == 1)[0])
            X = X_temp
            y = y_temp

        # show the statistic
        self.utils.data_description(X=X, y=y)

        if anomaly_stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=True, stratify=y)
        else:
            idx_normal = np.where(y == 0)[0]
            idx_abnormal = np.where(y == 1)[0]
            np.random.shuffle(idx_normal)

            train_size = int((1-self.test_size)*len(X))
            test_frac = self.test_size + 0.1
            while train_size >= len(idx_normal):
                train_size = int((1-test_frac)*len(X))
                test_frac += 0.1

            train_idx = idx_normal[:train_size]
            test_idx = np.concatenate([idx_normal[train_size:], idx_abnormal])

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

        # minmax scaling
        if minmax:
            scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        return {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}

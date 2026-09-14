import os

import torch
from pyexpat import features
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.anomalies.datasets.generator import DataGenerator
from src.model.iced import ICED
import logging
import sklearn.metrics
import wandb

logging.basicConfig(level=logging.WARNING)
import pandas as pd
import itertools
from tqdm import tqdm
import time
import numpy as np


def dataset_filter(data_generator, anomaly_stratify, generate_duplicates=True, n_samples_threshold=1000, max_dim=50):
    dataset_list_org = list(itertools.chain(data_generator.generate_dataset_list()))[0]

    dataset_list = []
    for dataset in dataset_list_org:
        add = True
        data_generator.seed = 0
        data_generator.dataset = dataset
        data = data_generator.generator(anomaly_stratify=anomaly_stratify)

        if not generate_duplicates and len(data['y_train']) + len(data['y_test']) < n_samples_threshold:
            add = False

        #if data['X_train'].shape[1] > max_dim:
        #    add = False

        if add:
            dataset_list.append(dataset)
        else:
            print(f"remove the dataset {dataset}")

    return dataset_list

def run_evaluate(clf, dataset=None, seedlist=None, model_name="ICED", generate_duplicates=True,
                 n_samples_threshold=1000, anomaly_stratify=True, max_dim=100, run_mode=None):
    data_generator = DataGenerator(generate_duplicates=generate_duplicates,
                                   n_samples_threshold=n_samples_threshold)
    print("\n Start evaluation: ")
    datasets = None
    if seedlist is None:
        seedlist = [0]

    if dataset is None:
        dataset_list = dataset_filter(data_generator, anomaly_stratify, generate_duplicates, n_samples_threshold, max_dim)
        print("Datasets: ", dataset_list)
        X, y = None, None
    else:
        dataset_list = [i for i in range(len(dataset))]
        datasets = dataset
        X, y = None, None

    print(f'{len(dataset_list)} datasets')
    columns = [model_name]
    df_AUCROC = pd.DataFrame(data=None, index=dataset_list, columns=columns)
    df_time = pd.DataFrame(data=None, index=dataset_list, columns=columns)

    for seed in seedlist:
        for i, dataset in tqdm(enumerate(dataset_list)):
            data_generator.seed = seed
            data_generator.dataset = dataset

            if datasets is not None:
                X, y = datasets[i]
                X, y = X.numpy(), y.numpy()
                data_generator.dataset = None

            data = data_generator.generator(X=X, y=y, anomaly_stratify=anomaly_stratify)

            try:
                start_time = time.time()
                if "Eval" in clf.__class__.__name__:
                    clf.fit(data['X_train'], features=data['X_train'].shape[-1])
                else:
                    clf.fit(data['X_train'])
                end_time = time.time()
                time_fit = end_time - start_time

                # predicting score (inference)
                start_time = time.time()
                score_test = -clf.score_samples(data['X_test'])
                end_time = time.time()
                time_inference = end_time - start_time

                result = {}
                aucroc = sklearn.metrics.roc_auc_score(y_true=data['y_test'], y_score=score_test)
                result['aucroc'] = aucroc

                # K.clear_session()
                print(f"Model: {model_name}, AUC-ROC: {result['aucroc']}")

            except Exception as error:
                print(f'Error in model fitting. Model:{model_name}, Error: {error}')
                time_fit, time_inference = np.nan, np.nan
                result = {'aucroc': np.nan}
                pass

            print(f'\nCurrent experiment parameters: {dataset}, model: {model_name}, metrics: {result}, '
                  f'fitting time: {time_fit}, inference time: {time_inference}')

            column_name = f"{model_name}_{seed}"
            df_AUCROC.at[dataset, column_name] = round(100 * result['aucroc'], 2)
            df_time.at[dataset, column_name] = time_fit + time_inference

        mean_aucroc = df_AUCROC[column_name].mean()
        mode_name = run_mode if run_mode is not None else "real"
        wandb.log(
            {
                f"anomalies_{mode_name}_{anomaly_stratify}/aucroc_mean_{seed}": mean_aucroc,
            }
        )

    def calc_mean(df):
        seed_cols = [c for c in df.columns if c.startswith(f"{model_name}_")]
        df[model_name] = df[seed_cols].mean(axis=1).round(2)

    calc_mean(df_AUCROC)
    wandb.log({
        f"anomalies_{mode_name}_{anomaly_stratify}/aucroc_mean_all": df_AUCROC[model_name].mean()
    })
    df_AUCROC.to_csv("".join(["AUROC_", model_name, f"_{anomaly_stratify}", ".csv"]), index=True)


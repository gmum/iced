import random
import numpy as np
import pandas as pd
import wandb

from openml import datasets as openml_datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


IDS = {
    "protein": 40966,
    "fourier": 14,
    "biodeg": 1494,
    "steel": 1504,
    "stock": 841,
    "energy": 1472,
    "collins": 40971,
    "texture": 40499,
}

SAMPLES = {
    "protein": [20, 50, 100, 200, 500],
    "fourier": [20, 50, 100, 200, 500],
    "biodeg": [20, 50, 100, 200, 500],
    "steel": [20, 50, 100, 200, 500],
    "stock": [20, 50, 100, 200],
    "energy": [50, 100, 200],
    "collins": [100, 200],
    "texture": [50, 100, 200, 500],
}

DOWNSTREAM_MODELS = {
    "knn": lambda seed: KNeighborsClassifier(),
    "lr": lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
    "rf": lambda seed: RandomForestClassifier(random_state=seed),
    "mlp": lambda seed: MLPClassifier(max_iter=1000, random_state=seed),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_openml_dataset(openml_id):
    dataset = openml_datasets.get_dataset(openml_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    X = pd.DataFrame(X, columns=attribute_names)
    y = pd.Series(y, name=dataset.default_target_attribute)

    cat_features = [
        col for col, is_cat in zip(attribute_names, categorical_indicator) if is_cat
    ]
    num_features = [
        col for col, is_cat in zip(attribute_names, categorical_indicator) if not is_cat
    ]

    return X, y, cat_features, num_features


def filter_rare_classes(X, y, min_samples=10):
    counts = y.value_counts()
    valid = counts[counts >= min_samples].index
    mask = y.isin(valid)
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def make_preprocessor(cat_features, num_features):
    # Paper uses imputation + categorical-to-numeric + z-score normalization.
    # This uses ordinal encoding instead of LOO target encoding to avoid leakage and keep sklearn-only.
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])

    return Pipeline([
        ("columns", pre),
        ("scaler", StandardScaler()),
    ])


def stratified_subsample(X, y, n, seed):
    if n >= len(X):
        return X, y

    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=n,
        stratify=y,
        random_state=seed,
    )
    return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


def prepare_split(openml_id, n_real, seed):
    X, y, cat_features, num_features = load_openml_dataset(openml_id)
    X, y = filter_rare_classes(X, y, min_samples=10)

    n_test = min(len(X) // 2, 500)

    X_oracle, X_test, y_oracle, y_test = train_test_split(
        X,
        y,
        test_size=n_test,
        stratify=y,
        random_state=seed,
    )

    X_train, y_train = stratified_subsample(X_oracle, y_oracle, n_real, seed)

    #try:
    #    X_train, X_val, y_train, y_val = train_test_split(
    #        X_train,
    #        y_train,
    #        train_size=0.8,
    #        stratify=y_train,
    #        random_state=seed,
    #    )
    #except:
    #    X_train, X_val, y_train, y_val = train_test_split(
    #        X_train,
    #        y_train,
    #        train_size=0.8,
    #        random_state=seed,
    #    )

    preprocessor = make_preprocessor(cat_features, num_features)
    preprocessor.fit(X_train)

    X_train_np = preprocessor.transform(X_train)
    #X_val_np = preprocessor.transform(X_val)
    X_test_np = preprocessor.transform(X_test)

    # Stable integer label encoding.
    classes = sorted(y.unique())
    label_map = {cls: i for i, cls in enumerate(classes)}

    y_train_np = y_train.map(label_map).to_numpy()
    #y_val_np = y_val.map(label_map).to_numpy()
    y_test_np = y_test.map(label_map).to_numpy()

    return X_train_np, X_test_np, y_train_np, y_test_np


def flatten_generated(data_syn):
    """
    Supports either:
      {class_id: ndarray}
      {"class_0": ndarray, "class_1": ndarray}
      {step: {class_id: ndarray}} handled outside.
    """
    X_parts, y_parts = [], []

    for key, Xc in data_syn.items():
        if isinstance(key, str) and key.startswith("class_"):
            cls = int(key.replace("class_", ""))
        else:
            cls = int(key)

        X_parts.append(np.asarray(Xc))
        y_parts.append(np.full(len(Xc), cls, dtype=int))

    return np.concatenate(X_parts), np.concatenate(y_parts)


def generate_synthetic(model, X_train, y_train, n_syn=500, device="cuda", **gen_kwargs):
    """
    Expected generator API:
        model.generate(X_train, y_train, num_samples=n_syn, device=device, ...)
    It may return:
        {class_id: ndarray}
    or:
        {step: {class_id: ndarray}}
    """
    out = model.generate(
        X_train,
        y_train,
        num_samples=n_syn,
        device=device,
        **gen_kwargs,
    )

    first_value = next(iter(out.values()))

    if isinstance(first_value, dict):
        return out

    return {"final": out}


def evaluate_one_split(
    generator,
    dataset_name,
    n_real,
    seed,
    device="cuda",
    n_syn=500,
    gen_kwargs=None,
):
    gen_kwargs = gen_kwargs or {}

    X_train, X_test, y_train, y_test = prepare_split(
        IDS[dataset_name],
        n_real,
        seed,
    )

    rows = []
    try:
        generated_by_step = generate_synthetic(
            generator,
            X_train,
            y_train,
            n_syn=n_syn,
            device=device,
            **gen_kwargs,
        )
    except Exception as e:
        print(e)
        for clf_name in DOWNSTREAM_MODELS.keys():
            rows.append({
                "dataset": dataset_name,
                "n_real": n_real,
                "seed": seed,
                "step": "failed",
                "classifier": clf_name,
                "acc_baseline": np.nan,
                "acc_augmented": np.nan,
                "delta": np.nan,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_syn": 0,
            })

        return rows

    for step, syn_dict in generated_by_step.items():
        X_syn, y_syn = flatten_generated(syn_dict)

        X_aug = np.concatenate([X_train, X_syn], axis=0)
        y_aug = np.concatenate([y_train, y_syn], axis=0)

        for clf_name, clf_factory in DOWNSTREAM_MODELS.items():
            baseline = clf_factory(seed)
            augmented = clf_factory(seed)

            baseline.fit(X_train, y_train)
            augmented.fit(X_aug, y_aug)

            acc_base = balanced_accuracy_score(y_test, baseline.predict(X_test)) * 100
            acc_aug = balanced_accuracy_score(y_test, augmented.predict(X_test)) * 100

            rows.append({
                "dataset": dataset_name,
                "n_real": n_real,
                "seed": seed,
                "step": step,
                "classifier": clf_name,
                "acc_baseline": acc_base,
                "acc_augmented": acc_aug,
                "delta": acc_aug - acc_base,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_syn": len(X_syn),
            })

    return rows


def evaluate_augmentation(
    generator,
    dataset_name,
    device="cuda",
    seeds=range(10),
    n_syn=500,
    gen_kwargs=None,
):
    all_rows = []

    for n_real in SAMPLES[dataset_name]:
        for seed in seeds:
            set_seed(seed)

            rows = evaluate_one_split(
                generator=generator,
                dataset_name=dataset_name,
                n_real=n_real,
                seed=seed,
                device=device,
                n_syn=n_syn,
                gen_kwargs=gen_kwargs,
            )

            all_rows.extend(rows)

            for row in rows:
                wandb.log(row)

        results = pd.DataFrame(all_rows)

        summary = (
            results
            .groupby(["dataset", "n_real", "classifier", "step"])
            .agg(
                acc_baseline_mean=("acc_baseline", "mean"),
                acc_baseline_std=("acc_baseline", "std"),
                acc_augmented_mean=("acc_augmented", "mean"),
                acc_augmented_std=("acc_augmented", "std"),
                delta_mean=("delta", "mean"),
                delta_std=("delta", "std"),
            )
            .reset_index()
        )

        overall = (
            results
            .groupby(["classifier", "step"])
            .agg(
                baseline_mean=("acc_baseline", "mean"),
                augmented_mean=("acc_augmented", "mean"),
                delta_mean=("delta", "mean"),
                baseline_std=("acc_baseline", "std"),
                augmented_std=("acc_augmented", "std"),
                delta_std=("delta", "std"),
            )
            .reset_index()
        )

        wandb.log({
            f"{dataset_name}/results_table": wandb.Table(dataframe=results),
            f"{dataset_name}/summary_table": wandb.Table(dataframe=summary),
            f"{dataset_name}/overall_table": wandb.Table(dataframe=overall),
        })

    return results, summary

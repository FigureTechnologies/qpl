import copy
import os
from os.path import join
from pathlib import Path
import pickle
from typing import Dict, List

import dask
import numpy as np
import pandas as pd
import ray
from ray.util.dask import ray_dask_get
import xgboost as xgb

from dmnet import util

ray.init(ignore_reinit_error=True)
dask.config.set(scheduler=ray_dask_get)

FLAGS = ("tr", "val", "te")

MODEL_PATH = "/home/andrewhah/tmp/plgbm.p"
PRED_DIR = "/home/andrewhah/tmp/plpred_jdh.parquet"
dataset = Dict[str, pd.DataFrame]


def get_dmatrix_paths() -> Dict[str, str]:
    bucket = "gs://dmnet/pl_seed/campaign/10/dmatrix/"
    return {f: join(bucket, f"{f}_gbm.parquet") for f in FLAGS}


def load_data_tr() -> dataset:
    paths = get_dmatrix_paths()
    data = {k: util.read_parquet_dask(v) for k, v in paths.items()}
    return dask.compute(data)[0]


def make_dmatrix(data: pd.DataFrame) -> xgb.DMatrix:
    data = data.drop(["record_nb"], axis=1)
    X = data.drop("y", axis=1)
    y = data["y"]
    return xgb.DMatrix(X, y)


def fit(data: dataset) -> None:
    params = {
        "eta": 0.1,
        "max_leaves": 27,
        "min_child_weight": 876,
        "subsample": 0.8998,
        "colsample_bytree": 0.8205,
        "eval_metric": "auc"
    }

    dtrain = make_dmatrix(data["tr"])
    dvalid = make_dmatrix(data["val"])

    train_kwargs = {
        "params": params,
        "dtrain": dtrain,
        "evals_result": {},
        "evals": [(dtrain, "train"), (dvalid, "valid")]
    }
    train_params = {
        "num_boost_round": 20000,
        "early_stopping_rounds": 50,
        "verbose_eval": 100
    }

    clf = xgb.train(**train_kwargs, **train_params)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)


@ray.remote
def predict_partition(clf: xgb.Booster, input_path: str, output_dir: str, features_tr: List[str]) -> None:
    X = pd.read_parquet(input_path, columns=features_tr + ["record_nb"])
    X = X.set_index("record_nb")
    pred = clf.predict(
        xgb.DMatrix(X),
        iteration_range=(0, clf.best_iteration)
    )
    output: pd.DataFrame = pd.DataFrame(pred, index=X.index, columns=["pred"]).reset_index()
    output["pred"] = output["pred"].astype(np.float32)  # pylint: disable = E1136, E1137
    output_path = join(output_dir, os.path.basename(input_path))
    output.to_parquet(output_path)


def predict(
    features_tr: List[str],
    input_dir: str = "gs://andrew-scratch-bucket/tmp/2202A.parquet"
) -> None:
    features_tr = copy.deepcopy(features_tr)
    for c in ("record_nb", "y"):
        if c in features_tr:
            features_tr.remove(c)

    with open(MODEL_PATH, "rb") as f:
        clf: xgb.Booster = pickle.load(f)

    clf.set_param({"nthread": 1})
    Path(PRED_DIR).mkdir(parents=True, exist_ok=True)
    pq_paths = util.get_parquet_paths(input_dir)
    clf = ray.put(clf)
    features_tr = ray.put(features_tr)
    ray.get([
        predict_partition.remote(clf=clf, input_path=p, output_dir=PRED_DIR, features_tr=features_tr)
        for p in pq_paths
    ])


def main():
    data = load_data_tr()
    fit(data)
    predict(features_tr=data["tr"].columns.to_list())


if __name__ == "__main__":
    main()

import copy
import os
from typing import Any, Dict

import dask
import dask.dataframe as dd
import lightgbm as lgb
from ray.util.dask import ray_dask_get
from sklearn.model_selection import train_test_split
import xgboost as xgb

import dmnet.config.config as cfg
from dmnet import util


util.init_ray(distributed=False, ignore_reinit_error=True)
dask.config.set(scheduler=ray_dask_get)


def load_data():
    dmt = util.pickle_load("gs://dmnet/plcamp/campaign/test/model/dmatrix_transformer.p")
    load_cols = dmt.use_cols
    df = dd.read_parquet(cfg.merged_data_path_tr, gather_statistics=False, columns=load_cols).compute()
    df = df.set_index(cfg.idx_cols)
    return df


def tts(df):
    df = df.reset_index()
    df = df.loc[df["campaign"] != "1000"]
    df = df.set_index(cfg.idx_cols)
    X = df.drop(cfg.target_col, axis=1)
    y = df[cfg.target_col]
    X_tr, X_other, y_tr, y_other = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    X_val, X_te, y_val, y_te = train_test_split(X_other, y_other, test_size=0.5, stratify=y_other, random_state=0)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def fit_lgb(dmatrix):
    X_tr, X_val, _, y_tr, y_val, _ = dmatrix
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.2,
        "num_leaves": 8,
        "num_threads": os.cpu_count(),
        "force_col_wise": True,
        "deterministic": True,
        "seed": 0,
        "min_data_in_leaf": 1000
    }
    dtr = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val)
    clf = lgb.train(
        params=params,
        train_set=dtr,
        early_stopping_rounds=50,
        valid_sets=(dtr, dval),
        valid_names=("tr", "val"),
        verbose_eval=25,
        num_boost_round=10_000
    )
    return clf


def fit_xgb(dmatrix):
    X_tr, X_val, _, y_tr, y_val, _ = dmatrix
    hyperparams = copy.deepcopy(cfg.booster["hyperparams_prune"])

    if "balanced" in hyperparams:
        hyperparams.pop("balanced")

    hyperparams["scale_pos_weight"] = (y_tr == 0).sum()/(y_tr == 1).sum()

    dtrain = xgb.DMatrix(data=X_tr, label=y_tr)

    dval = xgb.DMatrix(data=X_val, label=y_val)

    params = copy.deepcopy(cfg.booster["default_params"])
    params["eval_metric"] = [cfg.booster["metric"][0]]
    params = {**params, **hyperparams}

    evals_result: Dict[Any, Any] = {}
    clf = xgb.train(
        params=params,
        dtrain=dtrain,
        evals_result=evals_result,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=50,
        num_boost_round=10000,
        verbose_eval=100
    )
    return clf

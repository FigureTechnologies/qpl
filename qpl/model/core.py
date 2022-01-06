import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Type, Union
from subprocess import Popen

import dask
from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
import pandas as pd
import numpy as np
import ray
from ray.util.dask import ray_dask_get
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

import cac_calibrate.binner as b
import dmnet.util as utils
import dmnet.data.features.prune as prn
import bqutil.core as bq

ray.init(ignore_reinit_error=True)
dask.config.set(scheduler=ray_dask_get)

PROJ_DIR = "gs://jdh-bucket/projects/qpl/"
SERV_LOC = "gs://andrew-scratch-bucket/tmp/2202A.parquet"

class ClassificationModelMgr(object):
    def __init__(self,
            data_dict: Dict,
            target_col: str,
            opt_hyper: bool = False,
            auto_start: bool = True
    ):
        self.target_col = target_col
        self.data_dict = data_dict
        self.Xtr = self.data_dict['tr'].drop(self.target_col,axis=1)
        self.ytr = self.data_dict['tr'][self.target_col]
        if 'vl' in self.data_dict.keys():
            self.Xvl = self.data_dict['vl'].drop(self.target_col, axis=1)
            self.yvl = self.data_dict['vl'][self.target_col]
            self.use_cv = False
        else:
            self.use_cv = True
      
        if auto_start:
            if opt_hyper:
                param_star = self.optimize_hyperparams()
                self.train_clf(hyperparams = param_star['reg_params'])
            else:
                self.train_clf(hyperparams=xgb_default_params())

    def hyperparams(self):
        params = dict()
        params['reg_params'] = xgb_param_space()
        params['fit_params'] = xgb_fit_parms()
        params['loss'] = lambda y, pred: -1 * roc_auc_score(y, pred)
        return params

    def tune_clf(self, params: Dict[str, Any]):
        clf = xgb.XGBClassifier(**params['reg_params'])
        clf.fit(self.Xtr, 
                self.ytr, 
                eval_set=[(self.Xtr, self.ytr), (self.Xvl, self.yvl)], 
                **params['fit_params']
        )
        if self.use_cv:
            cv_loss = cross_val_score(estimator=clf, X=self.Xtr, y=self.ytr, cv=5)
            loss = -1 * cv_loss.mean()
        else:
            pred = clf.predict_proba(self.Xvl)[:, 1]
            loss = params['loss'](self.yvl, pred)
        return {'loss': loss, 'status': STATUS_OK}

    def optimize_hyperparams(self):
        params = self.hyperparams()
        trials = Trials()
        fn = self.tune_clf
        result = fmin(fn=fn, space=params, algo=tpe.suggest, max_evals=50, trials=trials)
        best_params = space_eval(params, result)
        return best_params

    def train_clf(self, hyperparams: Dict) -> None:
        clf = xgb.XGBClassifier(**hyperparams)
        clf.fit(self.data_dict['tr'].drop(self.target_col, axis=1), self.data_dict['tr'][self.target_col])
        model_loc = os.path.join(PROJ_DIR, "models", "xgb_model.p")
        utils.pickle_dump(clf, model_loc)
        fi = pd.Series(
                clf.feature_importances_,
                index=self.data_dict['tr'].drop(self.target_col, axis=1).columns
        )
        fi_loc = os.path.join(PROJ_DIR, "outputs", "fi_xgb.csv")
        fi.to_csv(fi_loc)

    

    def predict_clf(self, X: pd.DataFrame) -> pd.Series:
        model_loc = os.path.join(PROJ_DIR, "models", "xgb_model.p")
        clf = utils.pickle_load(model_loc)
        y_hat = pd.Series(clf.predict_proba(X)[:, 1], index=X.index)
        return y_hat

    def true_and_preds(self, Xy: pd.DataFrame) -> pd.DataFrame:
        y_hat = self.predict_clf(Xy.drop(self.target_col, axis=1))
        y_true = Xy[self.target_col]
        true_and_preds = pd.concat([y_hat, y_true], axis=1).rename(columns={0: 'y_hat'})
        return true_and_preds

    def score_models(self, export_res:bool = True) -> Dict[str, Tuple]:
        scores = {}
        data_res = []
        for desmat in self.data_dict.keys():
            act_n_pred = self.true_and_preds(self.data_dict[desmat]).copy()
            act_n_pred['type'] = desmat
            data_res.append(act_n_pred)
            roc = roc_auc_score(act_n_pred[self.target_col], act_n_pred['y_hat'])
            avp = average_precision_score(act_n_pred[self.target_col], act_n_pred['y_hat'])
            scores[desmat] = (roc, avp)
        if export_res:
            export_path = os.path.join(PROJ_DIR, "outputs", "results_xgb.csv")
            pd.concat(data_res).to_csv(export_path)
        return scores

def xgb_param_space():
    return {'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.3, .05)),
            'max_depth': hp.choice('max_depth', np.arange(2, 10, 1)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 800, 1)),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.25, 1.0),
            'subsample': hp.uniform('subsample', 0.3, 0.8),
            'max_leaves': hp.choice('max_leaves', np.arange(2, 128, 2)),
            'gamma': hp.choice('gamma', np.arange(0, 10, 1)),
            'n_estimators': 25
            }

def xgb_fit_parms():
    return {
#            'objective': 'binary:logistic',
            'early_stopping_rounds': 100
            }

def xgb_default_params():
    return {'learning_rate':0.2,
            'n_estimators': 50,
            'max_depth': 7, 
            'max_leaves': 26,
            'min_child_weight': 157,
            'gamma': 2,
            'colsample_bytree': 0.6424834,
            'subsample': .773427
            }

def predict_serve() -> None:
    model_loc = os.path.join(PROJ_DIR, "models", "xgb_model.p")
    clf = utils.pickle_load(model_loc)
    clf = ray.put(clf)
    PRED_PATH = "/home/josephhurley/projects/qpl/qpl/results/serve_preds.parquet"
    Path(PRED_PATH).mkdir(parents=True, exist_ok=True)

    features_tr = pd.read_parquet("gs://jdh-bucket/projects/qpl/data/pruned_valid.parquet").columns.to_list()
    features_tr.remove('y')
    features_tr = ray.put(features_tr)

    pq_paths = utils.get_parquet_paths(SERV_LOC)
    ray.get([predict_partition.remote(clf=clf,
        input_path=p,
        output_dir=PRED_PATH,
        features_tr=features_tr)
        for p in pq_paths
    ])


@ray.remote
def predict_partition( 
        clf: xgb.XGBClassifier(), 
        input_path: str, 
        output_dir: str,
        features_tr: List[str]
) -> None:
    X = pd.read_parquet(input_path, columns = features_tr + ["record_nb"])
    X = X.set_index("record_nb")
    pred = clf.predict_proba(X)[:, 1]
    output: pd.DataFrame = pd.DataFrame(pred, index=X.index, columns=["pred"]).reset_index()
    output["pred"] = output["pred"].astype(np.float32)
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    output.to_parquet(output_path)


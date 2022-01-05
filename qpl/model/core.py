import os
from typing import Any, Dict, List, Sequence, Tuple, Type, Union
from subprocess import Popen

from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

import cac_calibrate.binner as b
import dmnet.util as utils
import dmnet.data.features.prune as prn
import bqutil.core as bq

PROJ_DIR = "gs://jdh-bucket/projects/qpl/"

class ClassificationModelMgr(object):
    def __init__(self,
            data_dict: Dict,
            target_col: str,
            opt_hyper: bool = False
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
       
        if opt_hyper:
            param_star = self.optimize_hyperparams()
            print (param_star)
            self.train_clf(hyperparams = param_star['reg_params'])
        else:
            self.train_clf(hyperparams=xgb_default_params())

    def hyperparams(self):
        params = dict()
        params['reg_params'] = xgb_param_space()
        params['fit_params'] = xgb_fit_parms()
        params['loss'] = lambda y, pred: roc_auc_score(y, pred)
        return params

    def tune_clf(self, params: Dict[str, Any]):
        clf = xgb.XGBClassifier(**params['reg_params'])
        clf.fit(self.Xtr, self.ytr, eval_set=[(self.Xtr, self.ytr)], **params['fit_params'])
        if self.use_cv:
            cv_loss = cross_val_score(estimator=clf, X=self.Xtr, y=self.ytr, cv=5)
            loss = -1 * cv_loss.mean()
        else:
            pred = clf.predict(self.Xvl)
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
        utils.pickle_dump(clf, f"gs://jdh-bucket/projects/fund_net/models/{self.model_type}_clf")
        fi = pd.Series(clf.feature_importances_, index=self.data_dict['tr'].drop(self.target_col, axis=1).columns)
        fi.to_csv(f'/home/josephhurley/projects/fund_net/fund_net/fi_{self.model_type}_clf.csv')

    def predict_clf(self, X: pd.DataFrame) -> pd.Series:
        clf = utils.pickle_load(f"gs://jdh-bucket/projects/fund_net/models/{self.model_type}_clf")
        y_hat = pd.Series(clf.predict(X), index=X.index)
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
            export_path = os.path.join(PROJ_DIR, 
            pd.concat(data_res).to_csv(f'/home/josephhurley/projects/fund_net/fund_net/results_{self.model_type}_clf.csv')
        return scores

def xgb_param_space():
    return {'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.3, .05)),
            'max_depth': hp.choice('max_depth', np.arange(2, 10, 1)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1)),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.25, 1.0),
            'subsample': hp.uniform('subsample', 0.3, 0.8),
            'max_leaves': hp.choice('max_leaves', np.arange(2, 128, 2)),
            'gamma': hp.choice('gamma', np.arange(0, 10, 1)),
            'n_estimators': 500
            }

def xgb_fit_parms():
    return {
            'eval_metric': 'mape',
            'early_stopping_rounds': 50
            }

def xgb_default_params():
    return {'learning_rate':0.15,
            'max_depth': 2, #5,
            'max_leaves': 42, #6,
            'min_child_weight': 5, #2,
            'gamma': 0, #2,
            'colsample_bytree': 0.79637, #0.6244,
            'subsample': 0.43719 #.78
            }


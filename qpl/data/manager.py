import os
from typing import Any, Dict, List, Sequence, Tuple, Type, Union
from subprocess import Popen

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import cac_calibrate.binner as b
import dmnet.util as utils
import dmnet.data.features.prune as prn
import bqutil.core as bq

IDX_COLS = ['record_nb']

class ModelInputMgr(object):
    """
    Manages the model inputs for the various models
    """

    def __init__(self,
            design_matrix: pd.DataFrame,
            target_col: Any,
            test_split: float = 0.2, #Of total size of data
            val_split: float = 0.2 #Of total size of data
        ):
        self.target_col = target_col
        self.idx_cols = IDX_COLS
        self.test_split = test_split
        self.val_split = val_split
        if set(self.idx_cols).issubset(set(design_matrix.columns)):
            self.des_mat = self.init_clean_desmat(design_matrix)
            self.tr_te_val_dict = self.split_train_val_test()
        else:
            raise RuntimeError("design_matrix needs appropriate columns for index.")


    def init_clean_desmat(self, desmat: pd.DataFrame) -> pd.DataFrame:
        desmat = desmat.set_index(self.idx_cols)
        return desmat

    def split_train_val_test(self) -> Dict:
        des_mat = self.des_mat.copy()
        X, y = des_mat.drop(self.target_col, axis=1), des_mat[self.target_col]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=self.test_split, stratify=y)
        res = {'te': pd.concat([Xte, yte], axis=1)}
        val_tr_split = 1 - self.val_split / (1 - self.test_split)
        Xtr, Xvl, ytr, yvl = train_test_split(Xtr, ytr, train_size=val_tr_split, stratify=ytr)
        res['vl'] = pd.concat([Xvl, yvl], axis=1)
        res['tr'] = pd.concat([Xtr, ytr], axis=1)
        return res

class DataCleaner(object):
    """
    This class takes as input a dictionary containing training and test (maybe validation) sets
    and cleans each dataframe in it--for training sets, it fits artifacts. For testing sets, it uses
    trained artifacts.
    """
    def __init__(self, data_dict: Dict, target_col: str, max_corr: float):
        self.max_corr = max_corr
        self.target_col = target_col
        self.data_dict = data_dict

    def fit_cov_pruner(self, min_std: float = 0):
        Xtr = self.data_dict['tr'].drop([self.target_col], axis=1).copy()

        cov_pruner = prn.CovariancePruner(max_corr=self.max_corr, min_std=min_std)
        cov_pruner.fit(Xtr)
        utils.pickle_dump(cov_pruner, "gs://jdh-bucket/projects/qpl/artifacts/cov_pruner.p")
   
    def clean_data_dict(self):
        self.fit_cov_pruner()
        cov_pruner = utils.pickle_load("gs://jdh-bucket/projects/qpl/artifacts/cov_pruner.p")
        res = {}
        for dataset in self.data_dict.keys():
            key_cols = [self.target_col] 
            prX = cov_pruner.transform(self.data_dict[dataset].drop(key_cols, axis=1)) 
            prXY = prX.merge(self.data_dict[dataset][key_cols], left_index=True, right_index=True)
            res[dataset] = prXY
             
        return res



import logging
import os
from os.path import join
from typing import List

import bqutil.core as bq
import dask
import dask.dataframe as dd
import pandas as pd
from ray.util.dask import ray_dask_get

import dmnet.config.config as cfg
import dmnet.data.features.main as fm
from dmnet import util

logger = logging.getLogger(__name__)
MAX_BQ_THREADS = 4
DEFAULT_IDX_COLS = ["record_nb"]
LABEL_COL = "population_group"
TARGET_ROWS_EXPECTED = 1_500_000
PROJECT = 'figure-development-data'
DATASET = 'data_science_staging'
EXP_TABLES = ['experian_main_2112A', 'experian_premier_2112A']
SEED_LOC = "gs://experian-archive-bureau"


def process_training_data() -> None:
    """
    Load experian training data: For each date, load a date group, and union the results.
    """
    #dask.config.set(scheduler=ray_dask_get)

    dates = ("0215", "0515", "0815", "1115")
    
    #gs_bucket = cfg.path["src_data_dir"]["seed"]
    gs_bucket = SEED_LOC
    #idx_cols = cfg.idx_cols
    idx_cols = ['record_nb', 'campaign']
    df = dd.concat(
        [_merge_date_group(date=d, gs_bucket=gs_bucket, idx_cols=idx_cols) for d in dates],
        axis=0
    )

    df = df.compute()
    #assert df.shape[0] == TARGET_ROWS_EXPECTED
    
    df = trim_to_srv_cols(df)
    df["record_nb"] = 1
    df["record_nb"] = df["record_nb"].cumsum().astype(int)
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    df = dd.from_pandas(df, npartitions=10)
    #df.to_parquet(cfg.merged_data_path_tr, overwrite=True)
    df.to_parquet("gs://jdh-bucket/projects/qpl/data/training.parquet", overwrite=True)


def get_valid_cols() -> List[str]:
    """
    Will return a set of valid columns since the training data has a superset of the 
    columns in the serving data
    """
    cols = []
    for tbl in EXP_TABLES:
        cols = cols + bq.get_columns(project=PROJECT, dataset=DATASET, table=tbl)
    return set(cols)

def trim_to_srv_cols(xs: pd.DataFrame) -> pd.DataFrame:
    """
    Does the actual paring down of columns
    """
    exp_cols = get_valid_cols()
    cols = list(set(xs.columns).intersection(exp_cols)) + ['y']
    return xs[cols].copy()

def _merge_date_group(
    date: str,
    gs_bucket: str,
    idx_cols: List[str]
) -> dd.DataFrame:
    """
    For a given date, load PremierAB, PremierCD, and SCORE, and merge them
    on record_nb.
    Parameters
    ----------
    date: Experian-type date like '0415'
    gs_bucket: Path where experian data lives.
    idx_cols: Index columns
    """
    def load_experian_csv_wrapper(suffix: str) -> dd.DataFrame:
        fname = f"A1805077_{date}_{suffix}.csv"
        path = join(gs_bucket, fname)
        return _load_experian_csv(path=path, idx_cols=idx_cols)

    dfs_src = [load_experian_csv_wrapper(p) for p in ("PremierAB", "PremierCD", "SCORE")]
    merged = dd.concat(dfs_src, axis=1, join="inner")
    
    #merged[cfg.target_col] = (merged[LABEL_COL] == "PL").astype(int)
    merged['y'] = (merged[LABEL_COL] == "PL").astype(int)
    merged = merged.drop(LABEL_COL, axis=1)
    merged = merged.reset_index(drop=True)
    return merged

def _load_experian_csv(path, idx_cols: List[str]) -> dd.DataFrame:
    df = dd.read_csv(path, delimiter="|", sample_rows=100_000, assume_missing=True)
    df.columns = df.columns.str.lower()
    legal_prefix = ("premier", "fic")
    feature_cols = [c for c in df.columns if c.startswith(legal_prefix)]
    bk_cols = idx_cols + [LABEL_COL]
    keep_cols = feature_cols + bk_cols
    keep_cols = df.columns.intersection(keep_cols).to_list()
    df = df[keep_cols]
    df["record_nb"] = df["record_nb"].astype(int)
    df = df.set_index("record_nb")
    return df

if __name__ == "__main__":
#    get_valid_cols()
    process_training_data()

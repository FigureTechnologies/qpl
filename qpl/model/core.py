import pandas as pd

import fund_net.data.core as fndc
import fund_net.model.core as fndm

if __name__=="__main__":
    df = pd.read_parquet("gs://jdh-bucket/projects/fund_net/data/initial_extract")

    mim = fndc.ModelInputMgr(design_matrix = df, target_col="app_amount")
    desmat_dict = mim.tr_te_val_dict

    USE_LOG = False

    data_clnr = fndc.DataCleaner(
            data_dict = desmat_dict,
            target_col = "app_amount",
            log_y = USE_LOG,
            max_corr = 0.8,
            bin_y = False 
    )

    clean_desmat = data_clnr.clean_data_dict()

    """
    mdl_mgr = fndm.ClassificationModelMgr(
            data_dict = clean_desmat,
            target_col = 'app_amount',
            model_type = 'gbm',
            opt_hyper = False
    )
    """

    mdl_mgr = fndm.RegressionModelMgr(
            data_dict = clean_desmat,
            target_col = 'app_amount', 
            model_type = 'gbm',
            opt_hyper = True,
            use_log = USE_LOG
    )

    res = mdl_mgr.score_models(export_res=True)
    print (res)

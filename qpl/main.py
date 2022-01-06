import pandas as pd

import qpl.data.core as qdc
import qpl.data.manager as qdm
import qpl.model.core as qmc

if __name__=="__main__":
    df = pd.read_parquet("gs://jdh-bucket/projects/qpl/data/training_ss_cols.parquet")

    mim = qdm.ModelInputMgr(
            design_matrix = df, 
            target_col="y",
            test_split=0.1,
            val_split=0.2
    )
    
    desmat_dict = mim.tr_te_val_dict

    data_clnr = qdm.DataCleaner(
            data_dict = desmat_dict,
            target_col = "y",
            max_corr = 0.65,
    )

    clean_desmat = data_clnr.clean_data_dict()

    mdl_mgr = qmc.ClassificationModelMgr(
            data_dict = clean_desmat,
            target_col = 'y',
            opt_hyper = True
    )

    res = mdl_mgr.score_models(export_res=True)
    print (res)

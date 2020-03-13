import numpy as np
import pandas as pd


def reduce_mem(df):
    # Date time cols
    dt_col_vnames = df.select_dtypes('datetime').columns.values

    # Object dtypes
    obj_col_vnames = list(df.select_dtypes("O").columns.values)
    obj_col_vnames = [obj_col for obj_col in obj_col_vnames if obj_col not in dt_col_vnames]

    if len(obj_col_vnames) > 0:
        df[obj_col_vnames] = df[obj_col_vnames].astype("category")

    # Numerical dtypes
    num_col_vnames = df.select_dtypes(np.number).columns.values
    if len(num_col_vnames) > 0:
        for col in num_col_vnames:
            if df[col].max() < np.finfo(np.float32).max:
                try:
                    if np.array_equal(df[col], df[col].astype(int)):
                        df[col] = df[col].pipe(pd.to_numeric, downcast="integer")
                    else:
                        df[col] = df[col].pipe(pd.to_numeric, downcast="float")
                except ValueError:
                    df[col] = df[col].pipe(pd.to_numeric, downcast="float")
    return df

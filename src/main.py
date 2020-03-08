#%%

import pandas as pd
import numpy as np
import vaex
from IPython.display import display
from vaex.ml.sklearn import Predictor
import lightgbm
from sklearn import model_selection as skl_ms
import gc


#%%
# generate regression dataset
df_pd = pd.DataFrame(np.random.randint(2, size=(10**7, 100), dtype=np.uint8))
gc.collect()

df_pd.columns = [str(vname) for vname in list(df_pd.columns.values)]
target_vname = "target"
df_pd.rename(columns={"99": target_vname}, inplace=True, errors="ignore")
features = [vname for vname in list(df_pd.columns.values) if vname != target_vname]
df_pd_train, df_pd_test = skl_ms.train_test_split(df_pd, test_size=0.3, random_state=123)
df_vaex = vaex.from_pandas(df_pd)
df_vaex_train, df_vaex_test = df_vaex.ml.train_test_split(test_size=0.3, verbose=False)

#%%
date_time_fmt = "%Y/%m/%d %H:%M:%S"
start_time = pd.to_datetime("now")
print("Start time:", start_time.strftime(date_time_fmt))

booster = lightgbm.LGBMRegressor()
# vaex_model = Predictor(model=booster, features=features, target=target_vname, prediction_name='pred')
# vaex_model.fit(df=df_vaex_train)
# df_train = vaex_model.transform(df_vaex_train)

finish_time = pd.to_datetime("now")
print("Finish time:", finish_time.strftime(date_time_fmt), "or", (finish_time-start_time).seconds//60, "minutes.")

#%%

# df_vaex_test = vaex_model.transform(df_vaex_test)
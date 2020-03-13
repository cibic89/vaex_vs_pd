#%%

import gc
import os

import numpy as np
import pandas as pd
# import vaex
# from IPython.display import display
# from vaex.ml.sklearn import Predictor
import xgboost as xgb
from sklearn import model_selection as skl_ms

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['MAX_BATCH_SIZE'] = '32'
import autokeras as ak


#%%
# generate regression dataset
df_pd = pd.DataFrame(np.random.randint(2, size=(10**6, 21), dtype=np.uint8))
gc.collect()

df_pd.columns = [str(vname) for vname in list(df_pd.columns)]
target_vname = "target"
df_pd.rename(columns={df_pd.columns[-1]: target_vname}, inplace=True, errors="ignore")
features = [vname for vname in list(df_pd.columns) if vname != target_vname]
df_pd_train, df_pd_test = skl_ms.train_test_split(df_pd, test_size=0.3, random_state=123)
X_train, y_train = df_pd_train[features], df_pd_train[target_vname]
X_test, y_test = df_pd_test[features], df_pd_test[target_vname]
# df_vaex = vaex.from_pandas(df_pd)
# df_vaex_train, df_vaex_test = df_vaex.ml.train_test_split(test_size=0.3, verbose=False)

#%%
date_time_fmt = "%Y/%m/%d %H:%M:%S"
start_time = pd.to_datetime("now")
print("Start time:", start_time.strftime(date_time_fmt))

xgb_clf = xgb.XGBClassifier(tree_method="hist", ntree_limit=100)
xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="logloss", early_stopping_rounds=1)
xgb_evals_result = xgb_clf.evals_result()
print("XGBoost logloss on test set:", xgb_evals_result)
#  XGBoost logloss on test set: {'validation_0': {'logloss': [0.689716]}}

del xgb_clf
gc.collect()

finish_time = pd.to_datetime("now")
print("Finish time:", finish_time.strftime(date_time_fmt), "or", (finish_time-start_time).seconds//60, "minutes.\n")

#%%
date_time_fmt = "%Y/%m/%d %H:%M:%S"
start_time = pd.to_datetime("now")
print("Start time:", start_time.strftime(date_time_fmt))

ak_clf = ak.StructuredDataClassifier(max_trials=3, loss='binary_crossentropy', objective="val_loss", seed=123)
ak_clf.fit(X_train, y_train, epochs=10, verbose=0)
print("AutoKeras logloss on test set:", ak_clf.evaluate(X_test, y_test))
# AutoKeras logloss on test set: [0.6931553467941284, 0.5006833]

finish_time = pd.to_datetime("now")
print("Finish time:", finish_time.strftime(date_time_fmt), "or", (finish_time-start_time).seconds//60, "minutes.\n")

#%%

# xgb_clf = xgb.XGBClassifier()
# vaex_model = Predictor(model=booster, features=features, target=target_vname, prediction_name='pred')
# vaex_model.fit(df=df_vaex_train)
# df_train = vaex_model.transform(df_vaex_train)

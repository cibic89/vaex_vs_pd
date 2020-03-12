#%%

import pandas as pd
import numpy as np
# import vaex
# from IPython.display import display
# from vaex.ml.sklearn import Predictor
import xgboost as xgb
import autokeras as ak
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
X_train, y_train = df_pd_train[features], df_pd_train[target_vname]
X_test, y_test = df_pd_test[features], df_pd_test[target_vname]
# df_vaex = vaex.from_pandas(df_pd)
# df_vaex_train, df_vaex_test = df_vaex.ml.train_test_split(test_size=0.3, verbose=False)

#%%
date_time_fmt = "%Y/%m/%d %H:%M:%S"
start_time = pd.to_datetime("now")
print("Start time:", start_time.strftime(date_time_fmt))

xgb_clf = xgb.XGBClassifier(objective="binary:logistic", tree_method="approx", eval_metric="logloss")
xgb_clf.fit(X_train, y_train, )
xgb_pred_y = xgb_clf.predict(X_test)
print(xgb_clf.evaluate(X_test, y_test))

finish_time = pd.to_datetime("now")
print("Finish time:", finish_time.strftime(date_time_fmt), "or", (finish_time-start_time).seconds//60, "minutes.")

#%%
date_time_fmt = "%Y/%m/%d %H:%M:%S"
start_time = pd.to_datetime("now")
print("Start time:", start_time.strftime(date_time_fmt))

ak_clf = ak.StructuredDataClassifier(max_trials=3)
ak_clf.fit(X_train, y_train)
ak_pred_y = ak_clf.predict(X_test)
print(ak_clf.evaluate(X_test, y_test))

# xgb_clf = xgb.XGBClassifier()
# vaex_model = Predictor(model=booster, features=features, target=target_vname, prediction_name='pred')
# vaex_model.fit(df=df_vaex_train)
# df_train = vaex_model.transform(df_vaex_train)

finish_time = pd.to_datetime("now")
print("Finish time:", finish_time.strftime(date_time_fmt), "or", (finish_time-start_time).seconds//60, "minutes.")
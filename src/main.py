#%%

import gc
import os
import sys

import pandas as pd
# import vaex
# from IPython.display import display
# from vaex.ml.sklearn import Predictor
import xgboost as xgb
from sklearn import model_selection as skl_ms
main_dir = os.path.normpath(os.getcwd()+os.sep+os.pardir)
sys.path.append(main_dir)  # Add the main directory to sys.path
from src.functions import data_prep as fu  # in order to import functions.py
from src.functions import ml_prep as mp  # in order to import functions.py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['MAX_BATCH_SIZE'] = '32'
import autokeras as ak

#%%
# Dataset
# df_pd = pd.DataFrame(np.random.randint(2, size=(10**6, 21), dtype=np.uint8))
df_pd = pd.read_csv("data/raw/creditcard.csv")
df_pd = df_pd.drop(columns=["Time"]).pipe(fu.reduce_mem)
gc.collect()

# df_pd.columns = [str(vname) for vname in list(df_pd.columns)]
target_vname = "Class"
# df_pd.rename(columns={df_pd.columns[-1]: target_vname}, inplace=True, errors="ignore")
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


# ML data prep
X_train_xgb, X_test_xgb = mp.dmatricise(X_train, y_train), mp.dmatricise(X_test, y_test)

params = {
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}
xgb_clf = xgb.XGBClassifier()
xgb_param = xgb_clf.get_xgb_params()
cvresult = xgb.cv(params, X_train_xgb, num_boost_round=100, nfold=10, stratified=True,
       verbose_eval=False, early_stopping_rounds=5)

xgb_evals_result = dict()
xgb_clf = xgb.train(params, X_train_xgb, evals=[(X_train_xgb, "train"), (X_test_xgb, "test")],
                    num_boost_round=cvresult.shape[0], verbose_eval=False, early_stopping_rounds=5,
                    evals_result=xgb_evals_result)
print("XGBoost logloss on test set:", xgb_clf.best_score)
#  XGBoost logloss on test set: 'logloss': 0.002031

# del xgb_clf, X_train_xgb, X_test_xgb
# gc.collect()

finish_time = pd.to_datetime("now")
print("Finish time:", finish_time.strftime(date_time_fmt), "or", (finish_time-start_time).seconds//60, "minutes.\n")

#%%
date_time_fmt = "%Y/%m/%d %H:%M:%S"
start_time = pd.to_datetime("now")
print("Start time:", start_time.strftime(date_time_fmt))

ak_clf = ak.StructuredDataClassifier(max_trials=100, loss='binary_crossentropy', objective="val_loss", seed=123)
ak_clf.fit(X_train, y_train, epochs=1000, verbose=0)
print("AutoKeras logloss on test set:", ak_clf.evaluate(X_test, y_test))
# AutoKeras logloss on test set:

finish_time = pd.to_datetime("now")
print("Finish time:", finish_time.strftime(date_time_fmt), "or", (finish_time-start_time).seconds//60, "minutes.\n")

#%%

# xgb_clf = xgb.XGBClassifier()
# vaex_model = Predictor(model=booster, features=features, target=target_vname, prediction_name='pred')
# vaex_model.fit(df=df_vaex_train)
# df_train = vaex_model.transform(df_vaex_train)

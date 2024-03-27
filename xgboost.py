#Basic modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#to process the data and transform
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, minmax_scale
#To make pipeline
from sklearn.pipeline import Pipeline, make_pipeline
#Algo
from xgboost import XGBRegressor
#For cross validation & Hyperparameter tuning
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
#for model evaluation for regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#Now read the data, define, x & y then x_train, y_train, x_test, y_test

#splitting the data
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
xgb_model=XGBRegressor()
xgb_model.fit(x_train, y_train)
y_pred=xgb_model.predict(x_test)
y_train_pred=xgb_model.predict(x_train)

print(f"R2 score for training set: {(r2_score(y_train_pred, y_train))*100}")
print(f"R2 score for test set: {(r2_score(y_pred, y_test))*100}")


#Now do the Hyper parameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'random_state': [42],
    'min_split_loss':[1],
    'reg_lambda':[2, 4],
    'reg_alpha':[2, 5]
}

tuned_xgb_model=GridSearchCV(estimator=xgb_model, param_grid=param_grid, verbose=2)

tuned_xgb_model.fit(x_train, y_train)

tuned_xgb_model.best_params_
tuned_xgb_model.best_score_

#get the best parameters
tuned_xgb_model.best_params_

#update the model according to the best parameters
xgb_model = XGBRegressor(colsample_bytree=0.8,
                         learning_rate=0.1,
                         max_depth=7,
                         min_split_loss=1,
                         n_estimators=100,
                         random_state=42,
                         reg_alpha=2,
                         reg_lambda=4,
                         subsample=1.0)

xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_test)
y_train_pred = xgb_model.predict(x_train)

print(f"R2 score for training set: {r2_score(y_train, y_train_pred) * 100}")
print(f"R2 score for test set: {r2_score(y_test, y_pred) * 100}")

print(f"MAE for training set: {mean_absolute_error(y_train, y_train_pred)}")
print(f"MAE for test set: {mean_absolute_error(y_test, y_pred)}")



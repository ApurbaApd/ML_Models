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
from sklearn.ensemble import AdaBoostRegressor
#For cross validation & Hyperparameter tuning
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
#for model evaluation for regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#Now read the data, define, x & y then x_train, y_train, x_test, y_test

#splitting the data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
adb_model=AdaBoostRegressor()
adb_model.fit(x_train, y_train)
y_pred=adb_model.predict(x_test)
y_train_pred=adb_model.predict(x_train)

print(f"R2 score for training set: {(r2_score(y_train_pred, y_train))*100}")
print(f"R2 score for test set: {(r2_score(y_pred, y_test))*100}")

#do the hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
param_grid = {
    'n_estimators': [50,100],
    'learning_rate': [0.01, 0.05, 0.1, 1],
    'loss':['linear', 'square', 'exponential'],
    'random_state': [42],
   
}

tuned_adb_model=GridSearchCV(estimator=adb_model, param_grid=param_grid, verbose=2)

tuned_adb_model.fit(x_train, y_train)

tuned_adb_model.best_score_

#get the best parameters
tuned_adb_model.best_params_

#update the model using best parameters
adb_model=AdaBoostRegressor(learning_rate= 1, loss= 'square', n_estimators= 50, random_state=42)
adb_model.fit(x_train, y_train)
y_pred=adb_model.predict(x_test)
y_train_pred=adb_model.predict(x_train)

print(f"R2 score for training set: {(r2_score(y_train_pred, y_train))*100}")
print(f"R2 score for test set: {(r2_score(y_pred, y_test))*100}")


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
from sklearn.ensemble import RandomForestRegressor
#For cross validation & Hyperparameter tuning
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
#for model evaluation for regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#Now read the data, define, x & y then x_train, y_train, x_test, y_test

#splitting the data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
rf_model=RandomForestRegressor()
rf_model.fit(x_train, y_train)
y_pred=rf_model.predict(x_test)
y_train_pred=rf_model.predict(x_train)

print(f"R2 score for training set: {(r2_score(y_train_pred, y_train))*100}")
print(f"R2 score for test set: {(r2_score(y_pred, y_test))*100}")


#Do hyper parameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
param_grid={
    'n_estimators':[100],
    'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    'max_depth':[5],
    'min_samples_split':[2, 5, 10],
    'max_features':['sqrt', 'log2', None],
    'random_state':[42],
    
    ##And many other parameters
    
    
}

tuned_rf_model=GridSearchCV(estimator=rf_model, param_grid=param_grid, verbose=2)

tuned_rf_model.fit(x_train, y_train)

tuned_rf_model.best_params_
tuned_rf_model.best_score_

#Get the best parameters
tuned_rf_model.best_estimator_

#update the model using best parameters
rf_model=RandomForestRegressor(criterion='friedman_mse', max_depth=5, max_features=None,
                      random_state=42)
rf_model.fit(x_train, y_train)
y_pred=rf_model.predict(x_test)
y_train_pred=rf_model.predict(x_train)

print(f"R2 score for training set: {(r2_score(y_train_pred, y_train))*100}")
print(f"R2 score for test set: {(r2_score(y_pred, y_test))*100}")


print(f"MAE for training set:{(mean_absolute_error(y_train_pred, y_train))*100}")
print(f"MAE for test set:{(mean_absolute_error(y_pred, y_test))*100}")
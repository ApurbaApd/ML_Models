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
from sklearn.svm import SVR
#For cross validation & Hyperparameter tuning
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
#for model evaluation for regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#Now read the data, define, x & y then x_train, y_train, x_test, y_test

#splitting the data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
#scaling of the data
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
#model creation
svr_model=SVR()
svr_model.fit(x_train_scaled, y_train)
y_pred_train = svr_model.predict(x_train_scaled)
y_pred_test = svr_model.predict(x_test_scaled)

# Calculate R2 score for training and testing sets
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"R2 score for training set: {r2_train * 100}")
print(f"R2 score for test set: {r2_test * 100}")

## Now doing hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
param_grid = {
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3],
    'gamma':['scale', 'auto'],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.3], 
}

tuned_svr_model=GridSearchCV(estimator=svr_model, param_grid=param_grid, verbose=2)

tuned_svr_model.fit(x_train_scaled, y_train)

tuned_svr_model.best_score_

#get the best params
tuned_svr_model.best_params_

#update the model according to the best params
svr_model=SVR(C= 10, degree= 2, epsilon= 0.1, gamma= 'scale', kernel= 'rbf')
svr_model.fit(x_train_scaled, y_train)
y_pred_train = svr_model.predict(x_train_scaled)
y_pred_test = svr_model.predict(x_test_scaled)

# Calculate R2 score for training and testing sets
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"R2 score for training set: {r2_train * 100}")
print(f"R2 score for test set: {r2_test * 100}")
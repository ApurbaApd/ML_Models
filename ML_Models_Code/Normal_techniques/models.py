#Loading of the data
## House Pricing dataset
#from sklearn.datasets import load_boston
from sklearn.datasets import fetch_openml
# Load Boston Housing dataset using fetch_openml
df = fetch_openml(name='boston', version=2)

#important modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Reading the data
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
dataset.head()

## Dividing the dataset into independent and dependent features
##Skip the last feature(Price) as it is a dependent feature
x = dataset.iloc[:, :-1]  # independent features (all rows, all columns except the last one)
y = dataset.iloc[:, -1]   # dependent feature (all rows, only the last column)

#Splitting of the data
#Test Size 0.33, means 33% is test data & and rest are train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


#Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#Cross Validation->Here we divide our train and test data in such a way that every combination
# will be taken by the model and whoever's accuracy is better, that enetire thing will be combined.

#model
lin_reg=LinearRegression()
lin_reg.fit(x_train, y_train)
#mean square error
mse=cross_val_score(lin_reg, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
mean_mse=np.mean(mse)
print(mean_mse) # it should go towards zero, means perfomance is better

#prediction on the test data
y_pred=lin_reg.predict(x_test)

#Model Evaluation
r2_score1 = r2_score(y_pred, y_test)


## Ridge Regression
from sklearn.linear_model import Ridge
#For Hyper parameter tuning
from sklearn.model_selection import GridSearchCV

#model
ridge_model=Ridge()

params={
    'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]
}

ridge_regressor=GridSearchCV(ridge_model, params, scoring='neg_mean_squared_error',cv=5)
#fitting tha data, after Hyper parameter tuning
ridge_regressor.fit(x_train,y_train)

#Best parameters
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

#prediction
y_pred=ridge_regressor.predict(x_test)

#Evaluation
from sklearn.metrics import r2_score
r2_score1 = r2_score(y_pred, y_test)

## Lasso Regression
from sklearn.linear_model import Lasso
#For Hyper parameter tuning
from sklearn.model_selection import GridSearchCV

#model
lasso=Lasso()
params={'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regressor=GridSearchCV(lasso, params, scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(x_train,y_train)

#Best Hyper paramerts
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

#prediction on test data
y_pred=lasso_regressor.predict(x_test)

#Evaluation
from sklearn.metrics import r2_score
r2_score1 = r2_score(y_pred, y_test)
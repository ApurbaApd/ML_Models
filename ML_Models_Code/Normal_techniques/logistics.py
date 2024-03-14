#All the important modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#To split the data
from sklearn.model_selection import train_test_split
#Preprocessing of the data
from sklearn.preprocessing import StandardScaler, OneHotEncoder, minmax_scale
#To make pipeline
from sklearn.pipeline import make_pipeline, Pipeline
#ML Algos
from sklearn.linear_model import LinearRegression, Lasso, Ridge,LassoCV, RidgeCV,LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
#For cross validation
from sklearn.model_selection import cross_val_score
#For Hyper parameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#for model evaluation
#for the classification problem
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#For regression problem
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#loading the data
from sklearn.datasets import load_breast_cancer
#read the data
df=load_breast_cancer()
df

#Independent features(inputs)
x = pd.DataFrame(df['data'], columns=df['feature_names'])
x.head()
#Dependent features
y= pd.DataFrame(df['target'], columns=['Target'])
y
# Check y is balanced or imbalanced
y['Target'].value_counts() # Its a balanced

#splitting the data
#Test Size 0.33, means 33% is test data & and rest are train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#model creation
model1 = LogisticRegression(C=100, max_iter=100)

#Hyper parameter tuning
params=[{'C': [1, 5, 10]}, {'max_iter': [100, 150]}]
model = GridSearchCV(model1,  param_grid=params, scoring='f1', cv=5)
model.fit(x_train, y_train)

#Best params score & Best score
print(model.best_params_)
print(model.best_score_)

#According to the best params update the model
model = LogisticRegression(C=100, max_iter=150)
model.fit(x_train, y_train)

#Prediction
y_pred= model.predict(x_test)
y_pred


##For the evaluation of the model, Apply Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
confusion_matrix(y_pred, y_test)
## Accuracy is 97%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
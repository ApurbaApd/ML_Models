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

#All the regression models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB

#For cross validation & Hyperparameter tuning
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV

#for model evaluation for regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#for model evaluation for classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#load the data and read
from sklearn.datasets import load_breast_cancer



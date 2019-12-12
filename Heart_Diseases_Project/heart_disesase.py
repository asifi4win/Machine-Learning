import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_rows', 500)

dataset = pd.read_csv('heart_disease_dataset.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 13].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 9:13])
X[:, 9:13] = imputer.transform(X[:, 9:13])

feature_input = pd.DataFrame(X)
feature_input.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
feature_input

#categorical data need to be Encoded 
#Chest Pain
// No need for label encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 2] = labelencoder.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

feature_input = pd.DataFrame(X)
feature_input.columns = ['taCP','AtaCP','NaCP ','AsymCP','age','sex','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
feature_input

#categorical data need to be Encoded 
#Resting ElectroCardioGraphic Curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [9])
X = onehotencoder.fit_transform(X).toarray()

feature_input = pd.DataFrame(X)
feature_input.columns = ['1','2','rstEcg3','1','2','3','CP4','age','sex','trestbps','chol','fbs','thalach','exang','oldpeak','slope','ca','thal']
feature_input

#categorical data need to be Encoded 
#Resting ElectroCardioGraphic Curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [16])
X = onehotencoder.fit_transform(X).toarray()

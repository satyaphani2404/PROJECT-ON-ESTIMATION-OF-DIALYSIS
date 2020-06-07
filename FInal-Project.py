# PROJECT ON ESTIMATION OF DIALYSIS
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import patsy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.utils import resample
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
df = pd.read_csv('chronic_kidney_disease_full.csv')
df.head()
df.columns

df.isnull().sum()
df_copy = df.dropna()
df_copy.shape
df.columns
df.info()
# droping some features
df.drop(['rbc','pc','sod','pot','pcv','wbcc','rbcc'], axis=1, inplace=True)
# data preprocessing
df.info()
df_v1 = df
df_v1.replace(np.nan, 0, inplace=True)
df_v1.info()
df_v1['pcc'].value_counts()
df_v1['pcc'].replace(0, 'present', inplace=True)
df_v1['pcc'].value_counts()
df_v1['ba'].value_counts()
df_v1['ba'].replace(0, 'present', inplace=True)
df_v1['ba'].value_counts()
df_v1['htn'].value_counts()
df_v1['htn'].replace(0, 'yes', inplace=True)
df_v1['htn'].value_counts()
df_v1['dm'].value_counts()
df_v1['dm'].replace(0, 'yes', inplace=True)
df_v1['dm'].value_counts()
df_v1['cad'].value_counts()
df_v1['cad'].replace(0, 'yes', inplace=True)
df_v1['cad'].value_counts()
df_v1['appet'].value_counts()
df_v1['appet'].replace(0, 'poor', inplace=True)
df_v1['appet'].value_counts()
df_v1['pe'].value_counts()
df_v1['pe'].replace(0, 'yes', inplace=True)
df_v1['pe'].value_counts()
df_v1['ane'].value_counts()
df_v1['ane'].replace(0, 'yes', inplace=True)
df_v1['ane'].value_counts()
df_v1['class'].value_counts()
# encoding categorical data
df_v1 = pd.get_dummies(data=df_v1, drop_first=True)
df_v1.columns
df_v1.drop('class_notckd', axis=1, inplace=True)
df_v1 ['class'] = df['class']
df_v1.columns
df_v1.info()
v1col_list = list(df_v1.columns)
# extraction of feature
v1_features = []
[v1_features.append(col) for col in v1col_list if col != 'class']
X = df_v1[v1_features]
y = df_v1['class']
poly = PolynomialFeatures(include_bias=False, degree=2)
X_poly = poly.fit_transform(X)
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state = 42)
# Feature Scaling
ss = StandardScaler()
ss.fit(X_train)
X_train_sc = ss.transform(X_train)
X_test_sc = ss.transform(X_test)
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_sc, y_train)
logreg.score(X_train_sc, y_train)
logreg.score(X_test_sc, y_test)
logreg.coef_
np.exp(0.11018577)
predictions = logreg.predict(X_test_sc)
# Making the Confusion Matrix and Accuracy_score¶
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, predictions)
cm
cm = pd.DataFrame(cm, columns=['Predicted Negative','Predicted Positive'], index=['Actual Negative','Actual Positive'])
cm
accuracy_score(y_test, predictions)
df['class'].value_counts()
df_v1['class'].value_counts()
df_v1.shape
df_v1['class'].value_counts()
df_v1_maj = df_v1[ df_v1['class'] == 'ckd' ]
df_v1_min = df_v1[ df_v1['class'] == 'notckd' ]
df_upsample = resample(df_v1_maj, replace = True, n_samples = 4850, random_state = 42)
df_upsample = pd.concat([df_upsample, df_v1_min])
df_upsample['class'].value_counts()
X = df_upsample[v1_features]
y = df_upsample['class']
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state = 42)
ss.fit(X_train)
X_train_sc = ss.transform(X_train)
X_test_sc = ss.transform(X_test)
logreg.fit(X_train_sc, y_train)
# Earlier score was 0.96666666666
logreg.score(X_train_sc, y_train)
# Earlier score was 0.96
logreg.score(X_test_sc, y_test)
predictions = logreg.predict(X_test_sc)
# Making the Confusion Matrix and Accuracy_score
from sklearn import metrics
metrics.accuracy_score(predictions,y_test)
cm = confusion_matrix(y_test, predictions)
cm
cm = pd.DataFrame(cm, columns=['Predicted Negative','Predicted Positive'], index=['Actual Negative','Actual Positive'])
cm
accuracy_score(y_test, predictions)
# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=4)
model_knn.fit(X_train,y_train)
Prediction = model_knn.predict(X_test)
metrics.accuracy_score(Prediction,y_test)
DecisionTreeClassifier¶
from sklearn.tree import DecisionTreeClassifier
model_dec = DecisionTreeClassifier()
model_dec.fit(X_train,y_train)
Prediction = model_dec.predict(X_test)
metrics.accuracy_score(Prediction,y_test)
# RandomForestClassifier¶
from sklearn.ensemble import RandomForestClassifier
model_random = RandomForestClassifier()
model_random.fit(X_train,y_train)
Prediction = model_random.predict(X_test)
metrics.accuracy_score(Prediction,y_test)
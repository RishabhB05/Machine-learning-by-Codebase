import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv("L1&L2Regularization/Melbourne_housing_FULL.csv")


# we only select a few column which is looking for predicting the price
cols = ['Price', 'Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

dataset = df[cols]
print(dataset.head())



# now we fill up the missiing values with 0
dataset[cols] = dataset[cols].fillna(0)
print(dataset.head())

# creating dummy encoding for categorical column
dataset = pd.get_dummies(dataset)
print(dataset.head())

# creating regression
x = dataset.drop('Price', axis=1)
y = dataset['Price']

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.2, random_state=2)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
pred = reg.predict(testx)

# L1 Regularization
from sklearn.linear_model import Lasso
lasso = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso.fit(trainx, trainy)

# L2 Regularization
from sklearn.linear_model import Ridge
ridge = linear_model.Ridge(alpha=50, max_iter=100, tol=0.1)
ridge.fit(trainx, trainy)
print("Linear Regression Score:", reg.score(testx, testy))
print("Lasso Regression Score:", lasso.score(testx, testy))
print("Ridge Regression Score:", ridge.score(testx, testy))
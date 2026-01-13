import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("LinearRegressionMultiplevariable/homeprices.csv")

# there is a missing value in the data
# we can fill it with the mean value of that column

import math
median_bedroom = math.floor(df.bedrooms.median())

# here we filled the missing value with median 
df.bedrooms.fillna(median_bedroom, inplace=True)
print(df)


# using regression to predict prices based on multiple variables
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
predicted_price = reg.predict([[3000, 3, 40]])
print("Predicted price for a house with 3000 sqft area, 3 bedrooms, and 40 years old:", predicted_price[0])
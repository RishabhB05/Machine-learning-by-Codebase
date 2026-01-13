import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model

df = pd.read_csv("spliting/carprices.csv")
print(df)

# car milege vs sell price graph
plt.figure(figsize=(10,6))
plt.xlabel('price')
plt.ylabel('mileage')

plt.scatter(df['Mileage'], df['Sell Price($)'], color='blue', marker='o')
plt.show()

# car age vs sell price graph
plt.figure(figsize=(10,6))
plt.xlabel('price')
plt.ylabel('age')
plt.scatter(df['Age(yrs)'], df['Sell Price($)'], color='blue', marker='o')
plt.show()

# create linear regression object
# use for predicting values
x= df[['Mileage', 'Age(yrs)']]
y= df['Sell Price($)']

from sklearn.model_selection import train_test_split
# spliting data into training and testing data
# this line will split 20% of data as testing data and rest as training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# now training the model that is regression object
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
pred = reg.predict(x_test)
print("Predicted values are:", pred)
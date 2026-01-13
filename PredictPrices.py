import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# first reading the data
df = pd.read_csv("homeprices.csv")
print(df)  # Display the DataFrame to understand its structure

# convert into a graph
plt.figure(figsize=(10,6))
plt.xlabel('Year')
plt.ylabel('Per Capita Income (in USD)')
plt.scatter(df['year'], df['per capita income (US$)'], color='blue', marker='o')
plt.show()


# create linear regression object
# use for predicting values

reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])
pred = reg.predict([[2020]])
print("Predicted per capita income for 2020:", pred[0])

# plotting the 2020 prediction in the homeprices graph


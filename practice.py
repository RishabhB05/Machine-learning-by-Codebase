# TEST YOUR UNDERSTANDING
import numpy as np

# 1. EXPLAIN THIS:
X = np.array([[1, 2000, 3, 20],
              [1, 2500, 4, 15],
              [1, 1800, 2, 30],
              [1, 3000, 4, 10]])

y = np.array([300000, 400000, 250000, 500000])

# What does this line do? EXPLAIN in your own words
theta = np.linalg.inv(X.T @ X) @ X.T @ y

print("Theta values:", theta)

# 2. NOW IMPLEMENT THE PREDICT FUNCTION:
def predict_price(area, bedrooms, age, theta):
    """
    Given theta = [bias, weight_area, weight_bedrooms, weight_age]
    Calculate: price = bias + weight_area*area + weight_bedrooms*bedrooms + weight_age*age
    """
    # YOUR CODE HERE - 2 lines max
    return theta[0] + theta[1]*area + theta[2]*bedrooms + theta[3]*age
    
# Test it
pred = predict_price(2800, 3, 25, theta)
print(f"Predicted price for 2800 sqft, 3 bedrooms, 25 years: ${pred:,.0f}")

# 3. COMPARE WITH SKLEARN:
from sklearn.linear_model import LinearRegression

X_no_bias = X[:, 1:]  # Remove the bias column (1s)
sklearn_model = LinearRegression()
sklearn_model.fit(X_no_bias, y)
sklearn_pred = sklearn_model.predict([[2800, 3, 25]])[0]

print(f"Sklearn prediction: ${sklearn_pred:,.0f}")
print(f"Your prediction: ${pred:,.0f}")
print(f"Difference: ${abs(pred - sklearn_pred):,.0f}")
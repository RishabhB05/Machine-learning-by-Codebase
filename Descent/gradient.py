import numpy as np 


# This code is fixed , in every code we use the same code structure for gradient descent
def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 1000
    learning_rate = 0.01 # Increased slightly to reach the answer faster
    n = len(x)
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        
        # Calculate the Cost (Mean Squared Error) - optional but helpful
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        
        # Calculate gradients (the direction of the slope)
        md = (-2/n) * sum(x * (y - y_predicted))
        bd = (-2/n) * sum(y - y_predicted)
        
        # Update weights (the "baby steps" downhill)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
        if i % 100 == 0:
            print(f"Iteration {i}: m {m_curr:.3f}, b {b_curr:.3f}, cost {cost:.3f}")

    print(f"\nFinal Result -> m: {m_curr}, b: {b_curr}")



x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
gradient_descent(x, y)
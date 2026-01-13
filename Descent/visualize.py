import numpy as np
import matplotlib.pyplot as plt

def visualize_gradient_descent():
    # Simple quadratic: J(theta) = theta^2
    theta_values = np.linspace(-10, 10, 100)
    cost = theta_values ** 2
    
    # Gradient descent simulations
    # 0.1: Good step, 0.5: Fast, 1.0: Oscillates, 1.1: Diverges
    learning_rates = [0.1, 0.5, 1.0, 1.1]
    plt.figure(figsize=(12, 8))
    
    for i, lr in enumerate(learning_rates):
        plt.subplot(2, 2, i+1)
        
        # Plot cost function (the "Valley")
        plt.plot(theta_values, cost, 'b-', label='Cost Function', alpha=0.5)
        
        # Simulate gradient descent
        theta = 8.0  # Start point on the right side of the valley
        history = [theta]
        
        for step in range(20):
            gradient = 2 * theta  # The derivative of theta^2 is 2*theta
            theta = theta - lr * gradient
            history.append(theta)
            
            # Stop if diverging too far to keep the plot readable
            if abs(theta) > 50:
                break
        
        # Plot GD path
        history = np.array(history)
        plt.plot(history, history**2, 'ro-', linewidth=2, markersize=4, label='GD Path')
        plt.title(f'Learning rate = {lr}')
        plt.xlabel('Theta (Weight)')
        plt.ylabel('Cost (Error)')
        plt.grid(True)
        plt.ylim(-5, 100) # Keep scale consistent to see divergence
        
        if abs(theta) > 10:
            plt.text(0, 50, 'DIVERGED or OSCILLATING!', fontsize=10, color='red', ha='center')

    plt.tight_layout()
    plt.show() # Shows the plot immediately
    print("Graph generated! Look for the red line to see the 'steps' down the valley.")

# RUN THIS NOW
visualize_gradient_descent()
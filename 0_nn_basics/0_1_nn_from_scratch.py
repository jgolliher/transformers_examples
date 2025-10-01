# =============================================================================
# Neural Network from Scratch: An Educational Script
# =============================================================================
#
# Purpose:
# This script demonstrates how to train both a simple linear model and a
# neural network "by hand" using Python and NumPy. The goal is to demystify
# the core concepts of model training, particularly the process of gradient
# descent, without relying on high-level libraries like TensorFlow or PyTorch.

# The training is done using numerical differentiation, which is a simple
# (though computationally inefficient) way to calculate gradients. We
# "wiggle" each weight to see how it affects the model's error and then
# adjust the weight in the direction that reduces error.

# Script Sections:
# 1. Part 1: Training a simple Linear Regression Model from scratch.
# 2. Part 2: Training a Neural Network with one hidden layer from scratch.
#
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# For reproducibility, we set a random seed. This ensures that the same
# random numbers are generated each time the script is run.
np.random.seed(42)

# =============================================================================
# PART 1: INCREMENTAL LEARNING WITH A SIMPLE LINEAR MODEL
# =============================================================================
#
# We'll start with a basic linear model (y = b0 + b1*x) to understand the
# core training loop before moving on to a more complex neural network.
#
# -----------------------------------------------------------------------------
# 1.1 - Data Preparation
# -----------------------------------------------------------------------------
print("--- Part 1: Linear Model ---")

# Define the true parameters of our linear data
# Our goal is for the model to learn these values on its own
TRUE_INTERCEPT = 2.2222
TRUE_SLOPE = 5.4675
NUM_INSTANCES = 20

# Generate toy data based on the true parameters
# X is our feature, and y is our target variable
X_linear = np.random.normal(size=NUM_INSTANCES)
y_linear = TRUE_INTERCEPT + TRUE_SLOPE * X_linear

# To handle the intercept (bias) term in our model, we add a column of ones
# to our feature matrix. This is a standard trick in linear algebra that allows
# us to treat the intercept as just another weight to be learend.
# The shape will be (20, 2), where the first column is all ones.
X_linear_with_bias = np.column_stack((np.ones(NUM_INSTANCES), X_linear))

# Plot the generated linear data to visualize it
plt.figure(figsize=(10, 6))
plt.scatter(X_linear, y_linear, label="Original Data")
plt.title("Part 1: Generated Linear Data")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# 1.2 - Sanity Check with Scikit-Learn
# -----------------------------------------------------------------------------
# Before we build our own, let's see what a standard library finds.
# This gives us a baseline to check our results against.
from sklearn import linear_model

# We set `fit_intercept=False` because we have already added the bias column
reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(X_linear_with_bias, y_linear)
print(f"Scikit-learn found coefficients: {reg.coef_}")
print(f"(Should be close to [{TRUE_INTERCEPT}, {TRUE_SLOPE}])\n")

# -----------------------------------------------------------------------------
# 1.3 - Manual Implementation of Linear Model Training
# -----------------------------------------------------------------------------

# --- Hyperparameters ---
# These are settings we choose to guide the learning process
LEARNING_RATE = 0.01
NUM_EPOCHS = 100  # An epoch is one full pass through the entire dataset
EPSILON = 1e-7  # A very small number for calculating numerical derivatives

# --- Weight Initialization ---
# We start with random values for our weights (intercept and slope)
# The shape is (2, 1) to allow for matrix multiplication with our input data
weights_linear = np.random.normal(size=2).reshape(2, 1)
print(f"Initial random weights:\n{weights_linear}\n")


# --- Loss Function ---
# This function calculates how wrong our model's predictions are
# We use Mean Squared Error (MSE), a common choice for regression
def calculate_squared_error(weights, single_X_instance, single_y_target):
    """Calculates the squared error for a single data point."""
    # Predict y_hat by matrix-multiplying the input with the weights
    y_hat = np.matmul(single_X_instance, weights).flatten()
    # Calculate the squared difference between the true y and the prediction
    error = (single_y_target - y_hat) ** 2
    return error


# --- The Training Loop ---
# This is where the learning happens. We loop through the data many times,
# adjusting the weights slightly with each step. This is know as
# Stochastic Gradient Descent (SGD) because we update the weights one
# instance at a time

for epoch in range(NUM_EPOCHS):
    for i in range(len(X_linear_with_bias)):
        # For each weight, we calculate its impact on the error. This is the
        # partial derivative, which we approximate numerically.

        # Partial derivative for the intercept weight (weights_linear[0])
        weights_plus_eps = weights_linear.copy()
        weights_plus_eps[0] += EPSILON
        error_plus = calculate_squared_error(
            weights_plus_eps, X_linear_with_bias[i], y_linear[i]
        )

        weights_minus_eps = weights_linear.copy()
        weights_minus_eps[0] -= EPSILON
        error_minus = calculate_squared_error(
            weights_minus_eps, X_linear_with_bias[i], y_linear[i]
        )

        # The numerical derivative (gradient) is the change in error over the change in weight
        gradient_intercept = (error_plus - error_minus) / (2 * EPSILON)

        # Partial derivative for the slope weight (weights_linear[1])
        weights_plus_eps = weights_linear.copy()
        weights_plus_eps[1] += EPSILON
        error_plus = calculate_squared_error(
            weights_plus_eps, X_linear_with_bias[i], y_linear[i]
        )

        weights_minus_eps = weights_linear.copy()
        weights_minus_eps[1] -= EPSILON
        error_minus = calculate_squared_error(
            weights_minus_eps, X_linear_with_bias[i], y_linear[i]
        )

        gradient_slope = (error_plus - error_minus) / (2 * EPSILON)

        # --- Weight Update Rule ---
        # We update the weights by taking a small step in the opposite
        # direction of the gradient. This is the core of gradient descent.
        weights_linear[0] = weights_linear[0] - LEARNING_RATE * gradient_intercept
        weights_linear[1] = weights_linear[1] - LEARNING_RATE * gradient_slope

print(f"Final learned weights after {NUM_EPOCHS} epochs:\n{weights_linear}\n")

# -----------------------------------------------------------------------------
# 1.4 - Visualize the Result
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X_linear, y_linear, label="Original Data")


# Generate the line based on our learned weights.
X_plot = np.linspace(min(X_linear), max(X_linear), 100)
y_plot = weights_linear[0] + weights_linear[1] * X_plot
plt.plot(X_plot, y_plot, color="red", linewidth=3, label="Learned Regression Line")

plt.title("Part 1: Linear Model Fit")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# PART 2: IMPLEMENTING A SIMPLE NEURAL NETWORK
# =============================================================================
#
# Now we'll tackle a non-linear problem. A simple linear model can't capture
# this relationship, but a neural network can by learning a more complex
# function.
#
# Our Network Architecture:
# - Input Layer: 2 nodes (one for the feature X, one for the bias)
# - Hidden Layer: 2 neurons with a sigmoid activation function
# - Output Layer: 1 node (the final prediction)
#
# -----------------------------------------------------------------------------
# 2.1 - Data Preparation
# -----------------------------------------------------------------------------
print("\n--- Part 2: Neural Network ---")

# Create some non-linear (quadratic) toy data
X_nonlinear = np.random.normal(size=NUM_INSTANCES)
y_nonlinear = TRUE_INTERCEPT + TRUE_SLOPE * X_nonlinear - 3 * X_nonlinear**2

# Add the bias column, just like before.
X_nonlinear_with_bias = np.column_stack((np.ones(NUM_INSTANCES), X_nonlinear))

# Plot the non-linear data.
plt.figure(figsize=(10, 6))
plt.scatter(X_nonlinear, y_nonlinear)
plt.title("Part 2: Generated Non-Linear Data")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# 2.2 - Manual Implementation of Neural Network Training
# -----------------------------------------------------------------------------

# --- Hyperparameters ---
LEARNING_RATE_NN = 0.01
NUM_EPOCHS_NN = 1000  # Non-linear relationships often require more training.

# --- Weight Initialization ---
# We now have two sets of weights to initialize

# weights_input_hidden: Connects the 2 input nodes to the 2 hidden neurons
# Shape: (2, 2)
weights_input_hidden = np.random.normal(size=4).reshape(2, 2)

# weights_hidden_output: Connects the 2 hidden neurons to the 1 output node
# Shape: (2, 1)
weights_hidden_output = np.random.normal(size=2).reshape(2, 1)

print(f"Initial random weights (Input -> Hidden):\n{weights_input_hidden}\n")
print(f"Initial random weights (Hidden -> Output):\n{weights_hidden_output}\n")

# --- Forward Pass and Error Function ---
# This function defines the entire process of passing data through the network
# to get a prediction and then calculating the error.


def forward_pass_and_error_nn(w_i_h, w_h_o, single_X_instance, single_y_target):
    """Performs a full forward pass and calculates squared error."""
    # 1. Hidden Layer Calculation
    # Matrix multiply input with the first set of weights
    hidden_layer_input = np.matmul(single_X_instance, w_i_h)
    # Apply the sigmoid activation function. This introduces non-linearity.
    hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))

    # 2. Output Layer Calculation
    # Matrix multiply the hidden layer's output with the second set of weights
    output = np.matmul(hidden_layer_output, w_h_o)

    # 3. Calculate Error
    error = (single_y_target - output) ** 2
    return float(error)


# --- The Training Loop ---
for epoch in range(NUM_EPOCHS_NN):
    for i in range(len(X_linear_with_bias)):
        current_X = X_nonlinear_with_bias[i]
        current_y = y_nonlinear[i]

        # --- Calculate gradients for weights_input_hidden ---
        grad_input_hidden = np.zeros_like(weights_input_hidden)
        for row in range(weights_input_hidden.shape[0]):
            for col in range(weights_input_hidden.shape[1]):
                # Wiggle the weight at (row, col)
                w_plus = weights_input_hidden.copy()
                w_plus[row, col] += EPSILON
                error_plus = forward_pass_and_error_nn(
                    w_plus, weights_hidden_output, current_X, current_y
                )

                w_minus = weights_input_hidden.copy()
                w_minus[row, col] -= EPSILON
                error_minus = forward_pass_and_error_nn(
                    w_minus, weights_hidden_output, current_X, current_y
                )

                # Calculate the numerical derivative and store it
                grad_input_hidden[row, col] = (error_plus - error_minus) / (2 * EPSILON)

        # --- Calculate gradients for weights_hidden_output ---
        grad_hidden_output = np.zeros_like(weights_hidden_output)
        for row in range(weights_hidden_output.shape[0]):
            for col in range(weights_hidden_output.shape[1]):
                # Wiggle the weight at (row, col)
                w_plus = weights_hidden_output.copy()
                w_plus[row, col] += EPSILON
                error_plus = forward_pass_and_error_nn(
                    weights_input_hidden, w_plus, current_X, current_y
                )

                w_minus = weights_hidden_output.copy()
                w_minus[row, col] -= EPSILON
                error_minus = forward_pass_and_error_nn(
                    weights_input_hidden, w_minus, current_X, current_y
                )

                # Calculate the numerical derivative and store it.
                grad_hidden_output[row, col] = (error_plus - error_minus) / (
                    2 * EPSILON
                )

        # --- Update All Weights ---
        # Now we apply the updates using the gradients we just calculated.
        weights_input_hidden -= LEARNING_RATE_NN * grad_input_hidden
        weights_hidden_output -= LEARNING_RATE_NN * grad_hidden_output

print(f"Training complete after {NUM_EPOCHS_NN} epochs.\n")
print(f"Final learned weights (Input -> Hidden):\n{weights_input_hidden}\n")
print(f"Final learned weights (Hidden -> Output):\n{weights_hidden_output}\n")

# -----------------------------------------------------------------------------
# 2.3 - Visualize the Result
# -----------------------------------------------------------------------------
# To plot the learned function, we pass a range of X values through the
# trained network and plot the predictions.

plt.figure(figsize=(10, 6))
plt.scatter(X_nonlinear, y_nonlinear, label="Original Data")

# Sort X values for a smooth plot line.
sorted_indices = np.argsort(X_nonlinear)
X_sorted = X_nonlinear[sorted_indices]
X_sorted_with_bias = X_nonlinear_with_bias[sorted_indices]

# Perform a forward pass on the sorted data with the final learned weights.
hidden_layer_input = np.matmul(X_sorted_with_bias, weights_input_hidden)
hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
final_predictions = np.matmul(hidden_layer_output, weights_hidden_output)

plt.plot(
    X_sorted,
    final_predictions,
    color="red",
    linewidth=3,
    label="Learned Neural Network Function",
)
plt.title("Part 2: Neural Network Fit")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# CONCLUSION
# =============================================================================
#
# This script shows that a neural network can automatically learn a non-linear
# relationship from data without being explicitly told the form of the
# function (e.g., that it was quadratic). This ability to approximate complex
# functions is what makes neural networks so powerful.
#
# =============================================================================

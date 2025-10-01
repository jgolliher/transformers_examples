# =============================================================================
# Simple Neural Network in PyTorch: An Educational Script
# =============================================================================
#
# Purpose
# This script demonstrates how to build, train, and evaluate the same simple
# neural network form our "from scratch" example, but this time using the
# PyTorch library. The goal is to show how a deep learning framework
# simplifies the process by automating gradient calculation (autograd),
# providing pre-built layers, and offering utilities for data handling.
#
# Key PyTorch Concepts Introduced:
# 1. torch.nn.Module: The base class for all neural network modules. Our
#   network will be a class that inherits from this.
# 2. torch.utils.data.Dataset: A class for wrapping our data.
# 3. torch.utils.data.DataLoader: An iterator that handles batching,
#   shuffling, and loading the data for training.
# 4. Loss Function (nn.MSELoss): A pre-built module to compute the mean
#   squared error.
# Optimizer (torch.optim.SGD): An algorithm that implements stochastic
#   gradient descent, automatically handling weight updates.
#
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# For reproducibility, we set random seeds for both NumPy and PyTorch.
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# 1. Data Preparation
# =============================================================================
# We'll generate the exact same non-linear data as in the "from scratch" script.

# --- Data Generation ---
print("--- 1. Data Preparation ---")
TRUE_INTERCEPT = 2.2222
TRUE_SLOPE = 5.4675
NUM_INSTANCES = 20

X_numpy = np.random.normal(size=NUM_INSTANCES)
y_numpy = TRUE_INTERCEPT + TRUE_SLOPE * X_numpy - 3 * X_numpy**2

# --- Convert to PyTorch Tensors
# PyTorch uses its own mutli-dimensional array object called a "tensor". It's
# similar to a NumPy array but has special properties, like the ability to track
# operations for automatic differentiation and run on GPUs.
# We need to rehsape our data to have a feature dimenson. The shape shoul be
# [number_of_instances, number_of_features]. In our case, [20, 1].
X_tensor = torch.from_numpy(X_numpy.astype(np.float32)).view(-1, 1)
y_tensor = torch.from_numpy(y_numpy.astype(np.float32)).view(-1, 1)

print(f"Shape of X_tensor: {X_tensor.shape}")
print(f"Shape of y_tensor: {y_tensor.shape}\n")


# --- Create a Custom Dataset ---
# A Dataset object in PyTorch is a class that knows how to get the total
# number of items (__len__) and how to retrieve a specific item (__getitem__).
# This standardizes the way we access our data.
class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# --- Create a DataLoader ---
# The DataLoader takes a Dataset and provides an iterator to loop over the
# data in batches. Batching is crucial for efficient training on larger datasets.
# `shuffle=True` ensures that the model sees data in a different order each epoch,
# which helps prevent overfitting and improves generalization.
BATCH_SIZE = 5
dataset = SimpleDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============================================================================
# 2. Model Definition
# =============================================================================
# Here we define our neural network architecture. It will have the same structure:
# Input(1) -> Linear(1->2) -> Sigmoid -> Linear(2->1) -> Output(1)

print("--- 2. Model Definition ---")


class SimpleNeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We define the layers of the network as attributes
        # nn.Linear(input_features, output_features) automatically handles
        # the weights and biases for a linear transformation (y = Mx + b)
        self.layer1 = torch.nn.Linear(
            in_features=1, out_features=2
        )  # Corresponds to our weights_input_hidden
        self.activation = torch.nn.Sigmoid()
        self.layer2 = torch.nn.Linear(
            in_features=2, out_features=1
        )  # Corresponds to our weights_hidden_output

    def forward(self, x):
        # The forward method defines the "forward pass" - how data flows
        # through the neural network from input to output. PyTorch automatically
        # builds the computation graph to later perform backpropagation.
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


# Instantiate the model
model = SimpleNeuralNet()
print("Model Architecture:")
print(model)
print("\n")

# =============================================================================
# 3. Training the Model
# =============================================================================
print("--- 3. Training ---")

# --- Hyperparameters and Setup ---
LEARNING_RATE = 0.01
NUM_EPOCHS = 1000

# Loss Function: Measures the difference between the model's output and true target
# Mean Squared Error is the standard choice for regression problems
loss_fn = torch.nn.MSELoss()

# Optimizer: Implements the weight update algorithm (e.g., SGD)
# We pass it the model's parameters (weights and biases) so it knows what
# to update, and we set the learning rate
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

# --- The Training Loop ---

for epoch in range(NUM_EPOCHS):
    total_loss, total_accuracy = 0.0, 0.0
    # TQDM creates a progress bar to keep track of model training
    with tqdm(
        dataloader, unit="batch", desc=f"Epoch: {epoch+1}/{NUM_EPOCHS} [TRAINING]"
    ) as pbar:
        for x_batch, y_batch in pbar:
            # 1. Forward pass: Compute predicted by by passing x to the model
            y_pred = model(x_batch)

            # 2. Compute loss
            loss = loss_fn(y_pred, y_batch)

            # 3. Zero gradients: Before the backward pass, we must zero out the
            #   gradients from the previous iteration
            optimizer.zero_grad()

            # 4. Backward pass (Backpropagation): PyTorch's autograd engine
            #   computes the gradient of the loss wiht respect to all model
            #   parameters (the `requires_grad=True` tensors). This is the step
            #   we did manually with numerical differentiation before.
            loss.backward()

            # 5. Update weights: The optimizer updates teh parameters using the
            #   computed gradients and the learning rate.
            optimizer.step()

            # Update the progress bar's postfix with the current batch's loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")

print("\nTraining complete.\n")

# =============================================================================
# 4. Visualize the Result
# =============================================================================
print("--- 4. Visualization ---")

# To plot the learned function, we need to get the model's predictions.
# We switch the model to evaluation mode. This is important for models
# that have layers like Dropout or BatchNorm, which behave differently
# during training and inference.
model.eval()

# We use `torch.no_grad()` to tell PyTorch not to calculate gradients,
# which makes the forward pass faster and uses less memory.
with torch.no_grad():
    predicted = model(X_tensor).detach().numpy()

# Plot the original data and the learned function.
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label="Original Data")
# We need to sort the values for plotting a continuous line.
sorted_indices = np.argsort(X_numpy)
plt.plot(
    X_numpy[sorted_indices],
    predicted[sorted_indices],
    color="red",
    linewidth=3,
    label="Learned PyTorch Function",
)
plt.title("Part 2: Neural Network Fit (PyTorch)")
plt.xlabel("X (Feature)")
plt.ylabel("y (Target)")
plt.legend()
plt.grid(True)
plt.show()

MLP


import torch
import torch.nn as nn
import torch.optim as optim


# 1. Create Tensors
# PyTorch tensors are similar to NumPy arrays, but they can also run on a GPU if available
x = torch.tensor([1.0, 2.0, 3.0])  # 1D tensor (vector)
y = torch.tensor([4.0, 5.0, 6.0])  # 1D tensor (vector)


print(f"x: {x}")
print(f"y: {y}")


# 2. Tensor Operations
# Perform basic arithmetic operations on tensors
z = x + y
print(f"z (x + y): {z}")


z = x * y
print(f"z (x * y): {z}")


# 3. Define a Simple Neural Network Model (MLP)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 5)  # First linear layer (input size 3 -> output size 5)
        self.fc2 = nn.Linear(5, 1)  # Second linear layer (input size 5 -> output size 1)
        self.relu = nn.ReLU()       # Activation function


    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = self.fc2(x)             # Output layer
        return x


# 4. Instantiate the Model
model = SimpleNN()


# 5. Define Loss Function and Optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer


# 6. Dummy Input and Target Data
# In this example, we're just using random data for the sake of demonstration
input_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # 3 samples, each with 3 features
target_data = torch.tensor([[10.0], [20.0], [30.0]])  # Corresponding targets


# 7. Training Loop (for 100 epochs)
epochs = 100
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients before backpropagation


    # Forward Pass
    outputs = model(input_data)


    # Calculate Loss
    loss = criterion(outputs, target_data)


    # Backward Pass
    loss.backward()


    # Update Parameters
    optimizer.step()


    # Print Loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")


# 8. Making Predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # No need to calculate gradients during inference
    predictions = model(input_data)
    print("\nPredictions:")
    print(predictions)

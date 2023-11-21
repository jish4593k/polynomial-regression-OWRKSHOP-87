import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Importing the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define a simple polynomial regression model using PyTorch
class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_size, degree):
        super(PolynomialRegressionModel, self).__init__()
        self.poly = nn.Polynomial(degree)
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x_poly = self.poly(x)
        return self.linear(x_poly)

# Instantiate the model, loss function, and optimizer
model = PolynomialRegressionModel(input_size=1, degree=4)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_tensor)

    # Compute the loss
    loss = criterion(y_pred, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Convert X to PyTorch tensor and apply the polynomial transformation
X_pred_tensor = torch.tensor([[6.5]], dtype=torch.float32)
X_pred_poly_tensor = model.poly(X_pred_tensor)

# Predicting 6.5 level result using Polynomial Regression
y_pred_poly = model(X_pred_poly_tensor)

# Convert predictions back to numpy arrays
X_pred_np = X_pred_tensor.detach().numpy()
y_pred_poly_np = y_pred_poly.detach().numpy()

# Visualizing the results of Polynomial Regression
plt.scatter(X, y, c='red')
plt.plot(X_pred_np, y_pred_poly_np, c='blue')
plt.title('Polynomial Regression')
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

print("Predicted Salary for 6.5 level using Polynomial Regression:", y_pred_poly_np[0])

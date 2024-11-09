import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from scipy.integrate import solve_ivp

class FunctionApproximator(nn.Module):
    def __init__(self):
        super(FunctionApproximator, self).__init__()
        self.input_layer = nn.Linear(1, 20)
        self.hidden_layers = nn.ModuleList([nn.Linear(20, 20) for _ in range(9)])
        self.output_layer = nn.Linear(20, 1)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        x = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        output = self.output_layer(x)
        return output

def compute_derivatives(inputs, model):
    grads1 = model(inputs)
    grads2 = torch.autograd.grad(grads1, inputs, grad_outputs=torch.ones_like(grads1), create_graph=True)[0]
    grads3 = torch.autograd.grad(grads2, inputs, grad_outputs=torch.ones_like(grads2), create_graph=True)[0]
    return grads1, grads2, grads3
def compute_function_values_from_derivatives(grads1, inputs):
    delta_x = inputs[1] - inputs[0]
    function_values = torch.cumsum(grads1, dim=0) * delta_x
    function_values = function_values.clone()
    function_values -= function_values.clone()[0]
    return function_values


def train_model(model, criterion, optimizer, inputs, epochs=400, save_path=None):
    initial_condition_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs.device)
    initial_condition_grad_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs.device)
    boundary_grad_value_inf = torch.tensor([[1.0]], requires_grad=False).to(inputs.device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        grads1, grads2, grads3 = compute_derivatives(inputs, model)

        function_values = compute_function_values_from_derivatives(grads1, inputs)

        residual = grads3 + 0.5 * function_values * grads2
        equation_loss = criterion(residual, torch.zeros_like(residual))

        initial_condition_loss = criterion(grads1[0], initial_condition_grad_value_0) + \
                                 criterion(grads2[0], initial_condition_value_0) + \
                                 criterion(grads1[-1], boundary_grad_value_inf)

        loss = equation_loss + initial_condition_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.8f}')

    if save_path:
        torch.save(model.state_dict(), save_path)

    return function_values, grads1, inputs.cpu().detach().numpy()


def generate_data(start, end, num_points_per_unit):
    num_points = int((end - start) * num_points_per_unit)
    x = torch.linspace(start, end, num_points).unsqueeze(1).requires_grad_(True)
    return x



import torch


def get_closest_indices(x, specific_x):
    if not isinstance(specific_x, torch.Tensor):
        specific_x = torch.tensor(specific_x, dtype=torch.float32)

    specific_x = specific_x.to(x.device)

    indices = []
    for val in specific_x:
        index = torch.argmin(torch.abs(x - val))
        indices.append(index.item())
    return indices

def solve_blasius_runge_kutta(x):
    def equations(t, y):
        return [y[1], y[2], -0.5 * y[0] * y[2]]

    y0 = [0, 0, 0.33206]
    sol = solve_ivp(equations, [x[0], x[-1]], y0, t_eval=x)
    return sol.y[0], sol.y[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FunctionApproximator().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
save_path = "model_weights2500.pth"

if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path))
    print("模型权重已加载。")

start = 0
end = 2500
num_points_per_unit = 100
inputs = generate_data(start, end, num_points_per_unit).to(device)

function_values, grads1, x = train_model(model, criterion, optimizer, inputs, epochs=1, save_path=save_path)

x = x.flatten()
print("model over")
true_function_values, true_grads1 = solve_blasius_runge_kutta(x)
print("math over")



plot_range = [0, 10]

indices = (x >= plot_range[0]) & (x <= plot_range[1])
x_filtered = x[indices]
function_values_filtered = function_values[indices]
grads1_filtered = grads1[indices]
true_function_values_filtered = true_function_values[indices]
true_grads1_filtered = true_grads1[indices]

plt.figure(figsize=(10, 6))
plt.plot(x_filtered, function_values_filtered.detach().cpu().numpy(), label='Predicted Function (NN)', color='blue')
plt.plot(x_filtered, true_function_values_filtered, label='True Function (Runge-Kutta)', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparison of Predicted and True Function')
plt.legend()
plt.savefig('blasius_function_comparison_filtered.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_filtered, grads1_filtered.detach().cpu().numpy(), label='Predicted Derivative (NN)', color='green')
plt.plot(x_filtered, true_grads1_filtered, label='True Derivative (Runge-Kutta)', color='orange', linestyle='--')
plt.xlabel('x')
plt.ylabel('f\'(x)')
plt.title('Comparison of Predicted and True Derivative')
plt.legend()
plt.savefig('blasius_derivative_comparison_filtered.png')
plt.show()

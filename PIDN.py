import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import time
import csv
from thop import profile


# 定义神经网络结构
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


# 自动微分计算多阶导数
# 将模型的输出作为一阶导数，通过积分计算函数值
def compute_derivatives(inputs, model):
    grads1 = model(inputs)
    delta_x = inputs[1] - inputs[0]
    function_values = torch.cumsum(grads1, dim=0) * delta_x
    function_values = function_values.clone()
    function_values -= function_values.clone()[0]
    grads2 = torch.autograd.grad(grads1, inputs, grad_outputs=torch.ones_like(grads1), create_graph=True)[0]
    grads3 = torch.autograd.grad(grads2, inputs, grad_outputs=torch.ones_like(grads2), create_graph=True)[0]
    return function_values, grads1, grads2, grads3


# 训练模型
def train_model(model, criterion, optimizer, inputs, epochs=400, save_path=None):
    initial_condition_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs.device)  # f(0) 的真实值
    initial_condition_grad_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs.device)  # f'(0) 的真实值
    boundary_condition_value_end = torch.tensor([[1.0]], requires_grad=False).to(inputs.device)  # f'(∞) 的真实值

    # 加载或创建 CSV 文件
    csv_file = "training_log.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epochs", "Training Time (s)", "FLOPs (NN)", "FLOPs (Runge-Kutta)"])

    start_time = time.time()

    # 计算 FLOPs for NN
    dummy_input = torch.randn(1, 1).to(inputs.device)
    flops_nn, _ = profile(model, inputs=(dummy_input,), verbose=False)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        optimizer.zero_grad()

        function_values, grads1, grads2, grads3 = compute_derivatives(inputs, model)

        # 计算微分方程的残差
        equation_loss = grads3 + 0.5 * function_values * grads2
        equation_loss = criterion(equation_loss, torch.zeros_like(equation_loss))

        # 初始条件损失
        initial_condition_loss = criterion(function_values[0], initial_condition_value_0) + \
                                 criterion(grads1[0], initial_condition_grad_value_0)

        # 边界条件损失
        boundary_condition_loss = criterion(grads1[-1], boundary_condition_value_end)

        # 总损失
        loss = equation_loss + initial_condition_loss + boundary_condition_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.8f}')

    end_time = time.time()
    training_time = end_time - start_time

    # 保存模型权重
    if save_path:
        torch.save(model.state_dict(), save_path)

    # 计算 FLOPs for Runge-Kutta Method
    def runge_kutta_flops(num_points):
        # 每个步骤包含 4 次主要计算（对于经典的 RK4），每次有 3 个方程
        return num_points * 4 * 3

    num_points = inputs.size(0)
    flops_runge_kutta = runge_kutta_flops(num_points)

    # 数学方法计算时间
    # math_start_time = time.time()
    # y_true, _ = true_solution(inputs.cpu().detach().numpy().flatten())
    # math_end_time = time.time()
    # math_solution_time = math_end_time - math_start_time

    # 将训练时间和 FLOPs 写入 CSV 文件
    with open(csv_file, mode="a", newline='') as file:
        writer = csv.writer(file)
        # writer.writerow([epochs, training_time, flops_nn, flops_runge_kutta, math_solution_time])


# 数据生成
def generate_data(start, end, num_points_per_unit):
    num_points = int((end - start) * num_points_per_unit)
    x = torch.linspace(start, end, num_points).unsqueeze(1).requires_grad_(True)
    return x


# 使用数学方法求解方程的函数
def true_solution(x):
    def fun(x, y):
        return np.vstack((y[1], y[2], -0.5 * y[0] * y[2]))

    def bc(ya, yb):
        return np.array([ya[0], ya[1], yb[1] - 1])

    # 确保 x 的长度与生成数据一致
    y = np.zeros((3, x.size))
    y[0, 0] = 0  # f(0) = 0
    y[1, 0] = 0  # f'(0) = 0
    y[2, 0] = 0.33206  # f''(0) = 0.33206

    sol = solve_bvp(fun, bc, x, y)
    return sol.sol(x)[0], sol.sol(x)[1]


# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FunctionApproximator().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
save_path = "model_weights.pth"  # 保存模型权重的路径

# 检查是否存在保存的权重文件，如果存在则加载
if os.path.exists(save_path):
    state_dict = torch.load(save_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if not ("total_ops" in k or "total_params" in k)}
    model.load_state_dict(state_dict)
    print("模型权重已加载。")

# 生成数据
start, end = 0, 10
num_points_per_unit = 100
inputs = generate_data(start, end, num_points_per_unit).to(device)

# 训练模型
train_model(model, criterion, optimizer, inputs, epochs=100, save_path=save_path)

# 测试和可视化
outputs, grads1, _, _ = compute_derivatives(inputs, model)
outputs = outputs.cpu().detach().numpy()
grads1 = grads1.cpu().detach().numpy()

# 数学方式求解的结果
x = inputs.cpu().detach().numpy().flatten()
y_true, y_true_derivative = true_solution(x)

# 计算误差指标
# mse = mean_squared_error(y_true, outputs)
# mae = mean_absolute_error(y_true, outputs)
# std_dev = np.std(y_true - outputs)
# r2 = r2_score(y_true, outputs)

# 打印误差指标
# print(f'Mean Squared Error: {mse:.8f}')
# print(f'Mean Absolute Error: {mae:.8f}')
# print(f'Standard Deviation of Error: {std_dev:.8f}')
# print(f'R² Score: {r2:.4f}')

# 绘制函数值比较图并保存
plt.figure()
plt.plot(x, y_true, label='Mathematical Solution (Runge-Kutta Method)', color='red', linestyle='--')
plt.plot(x, outputs, label='Predicted Function (PINNs)', color='blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparison of Mathematical Solution and PINNs Function Prediction')
plt.savefig('comparison_function_plot.png')  # 保存生成的图片
plt.show()

# 绘制一阶导数比较图并保存
plt.figure()
plt.plot(x, y_true_derivative, label='Mathematical First Derivative (Runge-Kutta Method)', color='red', linestyle='--')
plt.plot(x, grads1, label='Predicted First Derivative (PINNs)', color='green')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Comparison of Mathematical and Predicted First Derivative (PINNs)')
plt.savefig('comparison_derivative_plot.png')  # 保存生成的图片
plt.show()

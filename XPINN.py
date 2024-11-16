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
def compute_derivatives(inputs, model):
    outputs = model(inputs)
    grads1 = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    grads2 = torch.autograd.grad(grads1, inputs, grad_outputs=torch.ones_like(grads1), create_graph=True)[0]
    grads3 = torch.autograd.grad(grads2, inputs, grad_outputs=torch.ones_like(grads2), create_graph=True)[0]
    return outputs, grads1, grads2, grads3


# 训练模型
def train_model(models, criterion, optimizers, inputs, epochs=400, save_paths=None):
    initial_condition_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs[0].device)  # f(0) 的真实值
    initial_condition_grad_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs[0].device)  # f'(0) 的真实值
    boundary_condition_value_end = torch.tensor([[1.0]], requires_grad=False).to(inputs[-1].device)  # f'(∞) 的真实值

    # 加载或创建 CSV 文件
    csv_file = "../training_log.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epochs", "Training Time (s)", "FLOPs (NN)", "FLOPs (Runge-Kutta)"])

    start_time = time.time()

    # 计算 FLOPs for NN
    dummy_input = torch.randn(1, 1).to(inputs[0].device)
    total_flops_nn = 0
    for model in models:
        flops_nn, _ = profile(model, inputs=(dummy_input,), verbose=False)
        total_flops_nn += flops_nn

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        total_loss = 0
        for i, model in enumerate(models):
            optimizer = optimizers[i]
            optimizer.zero_grad()

            # 获取子区域的数据
            sub_inputs = inputs[i]
            outputs, grads1, grads2, grads3 = compute_derivatives(sub_inputs, model)

            # 计算微分方程的残差
            equation_loss = grads3 + 0.5 * outputs * grads2
            equation_loss = criterion(equation_loss, torch.zeros_like(equation_loss))

            # 初始条件损失（仅在第一个子区域中应用）
            if i == 0:
                initial_condition_loss = criterion(outputs[0], initial_condition_value_0) + \
                                         criterion(grads1[0], initial_condition_grad_value_0)
            else:
                initial_condition_loss = 0

            # 边界条件损失（每个子区域的最后一个点与下一个子区域的第一个点相匹配）
            if i < len(models) - 1:
                next_model = models[i + 1]
                next_sub_inputs = inputs[i + 1]
                next_outputs, _, _, _ = compute_derivatives(next_sub_inputs, next_model)
                boundary_condition_loss = criterion(outputs[-1], next_outputs[0])
            else:
                # 对于最后一个子区域，应用边界条件 f'(∞) = 1
                boundary_condition_loss = criterion(grads1[-1], boundary_condition_value_end)

            # 总损失
            loss = equation_loss + initial_condition_loss + boundary_condition_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.8f}')

    end_time = time.time()
    training_time = end_time - start_time

    # 保存模型权重
    if save_paths:
        for i, model in enumerate(models):
            torch.save(model.state_dict(), save_paths[i])

    # 计算 FLOPs for Runge-Kutta Method
    def runge_kutta_flops(num_points):
        # 每个步骤包含 4 次主要计算（对于经典的 RK4），每次有 3 个方程
        return num_points * 4 * 3

    num_points = sum([inp.size(0) for inp in inputs])
    flops_runge_kutta = runge_kutta_flops(num_points)

    # 将训练时间和 FLOPs 写入 CSV 文件
    with open(csv_file, mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epochs, training_time, total_flops_nn, flops_runge_kutta])


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
num_subdomains = 2
models = [FunctionApproximator().to(device) for _ in range(num_subdomains)]
criterion = nn.MSELoss()
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]
save_paths = [f"model_weights_subdomain_{i}.pth" for i in range(num_subdomains)]

# 检查是否存在保存的权重文件，如果存在则加载
for i, model in enumerate(models):
    if os.path.exists(save_paths[i]):
        state_dict = torch.load(save_paths[i], map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if not ("total_ops" in k or "total_params" in k)}
        model.load_state_dict(state_dict)
        print(f"模型权重已加载：子域 {i}")

# 生成数据
start, end = 0, 10
num_points_per_unit = 100
subdomain_length = 5
inputs = []
math_inputs = generate_data(start, end, num_points_per_unit)  # 数学计算的采样数据
# 为每个子域生成独立的神经网络采样数据
for i in range(num_subdomains):
    sub_start = start + i * subdomain_length
    sub_end = sub_start + subdomain_length
    inputs.append(generate_data(sub_start, sub_end, num_points_per_unit // (end - start) * subdomain_length).to(device))

# 训练模型
train_model(models, criterion, optimizers, inputs, epochs=500, save_paths=save_paths)

# 测试和可视化
outputs_list = []
grads1_list = []
for i, model in enumerate(models):
    sub_inputs = inputs[i]
    outputs, grads1, _, _ = compute_derivatives(sub_inputs, model)
    outputs_list.append(outputs.cpu().detach().numpy())
    grads1_list.append(grads1.cpu().detach().numpy())

# 将各个子区域的结果拼接起来
x_combined = np.concatenate([inp.cpu().detach().numpy().flatten() for inp in inputs])
outputs_combined = np.concatenate(outputs_list).flatten()  # 扁平化输出结果
grads1_combined = np.concatenate(grads1_list).flatten()  # 扁平化一阶导数

# 计算整个区间上的数学解（只计算一次）
x_math = math_inputs.cpu().detach().numpy().flatten()
y_true_combined, y_true_derivative_combined = true_solution(x_math)

# 打印各个拼接部分的形状
print(f"x_combined shape: {x_combined.shape}")
print(f"outputs_combined shape: {outputs_combined.shape}")
print(f"grads1_combined shape: {grads1_combined.shape}")
print(f"x_math shape: {x_math.shape}")
print(f"y_true_combined shape: {y_true_combined.shape}")
print(f"y_true_derivative_combined shape: {y_true_derivative_combined.shape}")

# 绘制函数值比较图并保存
plt.figure()
plt.plot(x_math, y_true_combined, label='Mathematical Solution (Runge-Kutta Method)', color='red', linestyle='--')
plt.plot(x_combined, outputs_combined, label='Predicted Function (XPINNs)', color='blue')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparison of Mathematical Solution and XPINNs Function Prediction')
plt.savefig('comparison_function_plot_xpinns.png')  # 保存生成的图片
plt.show()

# 绘制一阶导数比较图并保存
plt.figure()
plt.plot(x_math, y_true_derivative_combined, label='Mathematical First Derivative (Runge-Kutta Method)', color='red', linestyle='--')
plt.plot(x_combined, grads1_combined, label='Predicted First Derivative (XPINNs)', color='green')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Comparison of Mathematical and Predicted First Derivative (XPINNs)')
plt.savefig('comparison_derivative_plot_xpinns.png')  # 保存生成的图片
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 定义每个子区间的神经网络
class SubdomainFunctionApproximator(nn.Module):
    def __init__(self):
        super(SubdomainFunctionApproximator, self).__init__()
        self.input_layer = nn.Linear(1, 40)
        self.hidden_layers = nn.ModuleList([nn.Linear(40, 40) for _ in range(9)])
        self.output_layer = nn.Linear(40, 1)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        x = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        output = self.output_layer(x)
        return output
# 将整个区间 [0, 10] 分成多个子区间
def partition_domain(num_points, num_subdomains):
    x_total = torch.linspace(0, 10, num_points).unsqueeze(1).requires_grad_(True)
    subdomain_size = num_points // num_subdomains
    subdomains = [x_total[i * subdomain_size: (i + 1) * subdomain_size] for i in range(num_subdomains)]
    return subdomains
def compute_derivatives(inputs, model):
    inputs = inputs.to(next(model.parameters()).device)  # 确保输入和模型参数在相同设备
    if not inputs.requires_grad:
        inputs.requires_grad_(True)  # 确保 inputs 可以追踪梯度
    outputs = model(inputs)
    grads1 = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs).to(inputs.device), create_graph=True)[0]
    grads2 = torch.autograd.grad(grads1, inputs, grad_outputs=torch.ones_like(grads1).to(inputs.device), create_graph=True)[0]
    grads3 = torch.autograd.grad(grads2, inputs, grad_outputs=torch.ones_like(grads2).to(inputs.device), create_graph=True)[0]
    return outputs, grads1, grads2, grads3

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
    return sol.sol(x)[0]

# 修改后的训练函数，适用于每个子网络
def train_subdomain_model(model, criterion, optimizer, inputs, epochs=400, save_path=None, interface_points=None):
    # 初始条件
    initial_condition_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs.device)  # f(0) 的真实值
    initial_condition_grad_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs.device)  # f'(0) 的真实值
    initial_condition_grad2_value_0 = torch.tensor([[0.33206]], requires_grad=False).to(inputs.device)  # f''(0) 的真实值

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs, grads1, grads2, grads3 = compute_derivatives(inputs, model)

        # 计算微分方程的残差
        equation_loss = grads3 + 0.5 * outputs * grads2
        equation_loss = criterion(equation_loss, torch.zeros_like(equation_loss).to(inputs.device))

        # 初始条件损失（仅对第一个子区间）
        initial_condition_loss = 0
        if interface_points is None:  # 这是第一个子区间
            initial_condition_loss = criterion(outputs[0].unsqueeze(0), initial_condition_value_0) + \
                                     criterion(grads1[0].unsqueeze(0), initial_condition_grad_value_0) + \
                                     criterion(grads2[0].unsqueeze(0), initial_condition_grad2_value_0)

        # 区间接口的损失（用于子区间的连接处）
        interface_loss = 0
        if interface_points is not None:  # 这是后续子区间，需要和前一个子区间匹配
            interface_loss = criterion(outputs[0].unsqueeze(0), interface_points['value'].detach()) + \
                             criterion(grads1[0].unsqueeze(0), interface_points['grad'].detach())

        # 总损失
        loss = equation_loss + initial_condition_loss + interface_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.8f}')

    # 保存模型权重
    if save_path:
        torch.save(model.state_dict(), save_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_subdomains = 4  # 将域划分为 4 个子区间
subdomains = partition_domain(num_points=10000, num_subdomains=num_subdomains)

models = [SubdomainFunctionApproximator().to(device) for _ in range(num_subdomains)]
optimizers = [optim.Adam(model.parameters(), lr=0.0001) for model in models]
criterions = [nn.MSELoss() for _ in range(num_subdomains)]

# 对每个子区间进行训练
interface_data = None  # 第一个子区间没有接口数据
for i in range(num_subdomains):
    print(f"Training subdomain {i + 1}/{num_subdomains}")
    save_path = f"subdomain_model_{i + 1}.pth"
    if os.path.exists(save_path):
        models[i].load_state_dict(torch.load(save_path, map_location=device))
        print(f"Subdomain {i + 1} model weights loaded.")
    inputs = subdomains[i].to(device)

    # 如果不是第一个子区间，需要接口数据
    if i > 0:
        # 计算前一个子区间的输出作为接口数据
        prev_inputs = subdomains[i - 1][-1:].to(device)
        prev_inputs.requires_grad_(True)  # 确保 prev_inputs 可以追踪梯度
        prev_outputs, prev_grads1, _, _ = compute_derivatives(prev_inputs, models[i - 1])

        # 保留接口数据
        interface_data = {
            'value': prev_outputs.to(device),
            'grad': prev_grads1.to(device)
        }

    # 训练模型
    train_subdomain_model(models[i], criterions[i], optimizers[i], inputs, epochs=1000, save_path=save_path,
                          interface_points=interface_data)
# 测试和可视化所有子区间的解
all_outputs = []
with torch.no_grad():
    for i in range(num_subdomains):
        subdomain_outputs = models[i](subdomains[i].to(device)).cpu().detach().numpy()
        all_outputs.append(subdomain_outputs)
test_outputs = np.concatenate(all_outputs, axis=0)

# 数学方式求解的结果
x = torch.linspace(0, 10, 10000).numpy()
y_true = true_solution(x)

# 计算误差指标
mse = mean_squared_error(y_true, test_outputs)
mae = mean_absolute_error(y_true, test_outputs)
std_dev = np.std(y_true - test_outputs)
r2 = r2_score(y_true, test_outputs)

# 打印误差指标
print(f'Mean Squared Error: {mse:.8f}')
print(f'Mean Absolute Error: {mae:.8f}')
print(f'Standard Deviation of Error: {std_dev:.8f}')
print(f'R² Score: {r2:.4f}')

# 绘制结果并保存
plt.plot(x, y_true, label='True Solution (Mathematical)', color='red')
plt.plot(x, test_outputs, label='Predicted Function (XPINNs)', color='blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparison of True Solution and XPINNs Prediction')
plt.savefig('comparison_plot_xpinns.png')
plt.show()

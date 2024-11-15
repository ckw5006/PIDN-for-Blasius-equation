import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 定义神经网络结构
class FunctionApproximator(nn.Module):
    def __init__(self):
        super(FunctionApproximator, self).__init__()
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

# 自动微分计算多阶导数
def compute_derivatives(inputs, model):
    outputs = model(inputs)
    grads1 = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    grads2 = torch.autograd.grad(grads1, inputs, grad_outputs=torch.ones_like(grads1), create_graph=True)[0]
    grads3 = torch.autograd.grad(grads2, inputs, grad_outputs=torch.ones_like(grads2), create_graph=True)[0]
    return outputs, grads1, grads2, grads3

# 训练模型
def train_model(model, criterion, optimizer, inputs, epochs=400, save_path=None):
    initial_condition_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs.device)  # f(0) 的真实值
    initial_condition_grad_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs.device)  # f'(0) 的真实值
    initial_condition_grad2_value_0 = torch.tensor([[0.33206]], requires_grad=False).to(inputs.device)  # f''(0) 的真实值

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs, grads1, grads2, grads3 = compute_derivatives(inputs, model)

        # 计算微分方程的残差
        equation_loss = grads3 + 0.5 * outputs * grads2
        equation_loss = criterion(equation_loss, torch.zeros_like(equation_loss))

        # 初始条件损失
        initial_condition_loss = criterion(outputs[0], initial_condition_value_0) + \
                                 criterion(grads1[0], initial_condition_grad_value_0) + \
                                 criterion(grads2[0], initial_condition_grad2_value_0)

        # 总损失
        loss = equation_loss + initial_condition_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.8f}')

    # 保存模型权重
    if save_path:
        torch.save(model.state_dict(), save_path)

# 数据生成
def generate_data(num_points):
    x = torch.linspace(0, 10, num_points).unsqueeze(1).requires_grad_(True)  # 在区间 [0, 5] 上生成数据并设置requires_grad
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
    return sol.sol(x)[0]

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FunctionApproximator().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
save_path = "model_weights.pth"  # 保存模型权重的路径

# 检查是否存在保存的权重文件，如果存在则加载
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path))
    print("模型权重已加载。")

# 生成数据
num_points = 10000
inputs = generate_data(num_points).to(device)

# 训练模型
train_model(model, criterion, optimizer, inputs, epochs=100, save_path=save_path)

# 测试和可视化
with torch.no_grad():
    test_outputs = model(inputs).cpu().detach().numpy()

# 数学方式求解的结果
x = inputs.cpu().detach().numpy().flatten()
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
plt.plot(x, test_outputs, label='Predicted Function (NN)', color='blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparison of True Solution and NN Prediction')
plt.savefig('comparison_plot.png')  # 保存生成的图片
plt.show()

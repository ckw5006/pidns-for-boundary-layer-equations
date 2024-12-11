import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import solve_ivp

# 定义神经网络结构，接受两个输入维度：采样点本身数值和时间 t
class FunctionApproximatorWithTime(nn.Module):
    def __init__(self):
        super(FunctionApproximatorWithTime, self).__init__()
        self.input_layer = nn.Linear(1, 20)  # 输入层接受1个维度
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
def train_model(model, criterion, optimizer, inputs, beta, epochs=400, save_path=None):
    initial_condition_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs.device)  # f(0) 的真实值
    initial_condition_grad_value_0 = torch.tensor([[0.0]], requires_grad=False).to(inputs.device)  # f'(0) 的真实值
    boundary_grad_value_inf = torch.tensor([[1.0]], requires_grad=False).to(inputs.device)  # f'(∞) 的真实值
    # initial_condition_fpp_value = torch.tensor([[fpp_0]], requires_grad=False).to(inputs.device)  # f''(0) 的真实值

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"模型权重已加载: {save_path}")
    # 记录训练开始时间
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()


        # 使用新方法计算函数值
        function_values, grads1, grads2, grads3 = compute_derivatives(inputs, model)

        # Falkner-Skan 方程残差计算
        residual = grads3 + function_values * grads2 + beta * (1 - grads1 ** 2)

        equation_loss = criterion(residual, torch.zeros_like(residual))

        # 初始条件损失
        initial_condition_loss = criterion(grads1[0], initial_condition_grad_value_0) + \
                                 criterion(function_values[0], initial_condition_value_0) + \
                                 criterion(grads1[-1], boundary_grad_value_inf)
                                 # criterion(grads2[0], initial_condition_fpp_value)  # f''(0) 的初始条件

        # 总损失
        loss = equation_loss + initial_condition_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Beta: {beta}, Loss: {loss.item():.8f}')

        # 在最后一轮训练后打印 0 点处的二阶导数值
        if epoch == epochs - 1:
            print(f"Epoch {epoch + 1}: 二阶导数在0点处的数值为 {grads2[0].item():.8f}")

    # 记录训练结束时间
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练耗时: {training_time:.4f} 秒")

    # 保存模型权重
    # torch.save(model.state_dict(), save_path)
    # print(f"模型权重已保存: {save_path}")

    # 返回最后一轮的函数值、导数值及对应的输入点
    return function_values.detach().cpu().numpy(), grads1.detach().cpu().numpy(), inputs.cpu().detach().numpy()

# 数据生成，加入时间维度并进行归一化
def generate_data_with_time(start, end, num_points_per_unit):
    num_points = int((end - start) * num_points_per_unit)
    x = torch.linspace(start, end, num_points).unsqueeze(1).requires_grad_(True)
    return x

# 使用Runge-Kutta法解法兰克福方程
def solve_falkner_skan_runge_kutta(x, beta, fpp_0):
    def equations(t, y):
        return [y[1], y[2], -beta * (1 - y[1] ** 2) - y[0] * y[2]]

    y0 = [0, 0, fpp_0]
    sol = solve_ivp(equations, [x[0], x[-1]], y0, t_eval=x)
    return sol.y[0], sol.y[1]

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FunctionApproximatorWithTime().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义 beta 值列表及对应的 f''(0) 值
beta_values = [-0.16, 0.0, 1.0, 2.0]
fpp_0_values = [0.190, 0.4696, 1.2326, 1.6872]
beta_values = [-0.16, 0.0, 2.0]
fpp_0_values = [0.190, 0.4696, 1.6872]
beta_values = [-0.18, -0.19, -0.198]
beta_values = [-0.198, -0.19, -0.18, -0.16, -0.14, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0]

# beta_values = [0.0]
# fpp_0_values = [0.4696]

colors = ['blue', 'green', 'purple', 'orange']
# 生成数据
start = 0
end = 5
num_points_per_unit = 100
inputs = generate_data_with_time(start, end, num_points_per_unit).to(device)

# 保存函数值和导数值的字典
results = {}

# 定义感兴趣的点
specific_x_full = np.linspace(0, end, 500)
specific_x_subset = np.linspace(0, 5, 500)

for beta in beta_values:
    # 设置权重保存路径
    save_path = f"model_weights_with_time_beta_{beta}.pth"
    # 对于每一个 beta 值，都重新初始化模型和优化器
    model = FunctionApproximatorWithTime().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型并返回最终的函数值、导数值以及对应的输入点
    function_values, grads1, x = train_model(model, criterion, optimizer, inputs, beta, epochs=500,
                                             save_path=save_path)

    # 保存结果
    results[beta] = {
        "function_values": function_values,
        "grads1": grads1,
        "x": x[:, 0].flatten()  # 确保 x 是一维的
    }

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


# 定义 Blasius 方程的常微分方程系统
def blasius_ode(x, y):
    f, f_prime, f_double_prime = y
    return [f_prime, f_double_prime, -0.5 * f * f_double_prime]


# 射线法目标函数，用于调整 f''(0) 使得 f'(x) 在远端满足边界条件（f'(\infty) = 1）
def shooting_objective(f_double_prime_0, target_x=10):
    # 初始条件：f(0) = 0, f'(0) = 0, f''(0) = f_double_prime_0
    y0 = [0, 0, f_double_prime_0]

    # 使用 solve_ivp 进行积分计算
    sol = solve_ivp(blasius_ode, [0, target_x], y0, t_eval=[target_x], method='RK45')

    # 返回远端 f'(target_x) - 1，目标是让它接近 0
    return sol.y[1, -1] - 1


# 使用 root_scalar 查找合适的初值 f''(0)
sol = root_scalar(shooting_objective, bracket=[0.1, 1.0], method='bisect', xtol=1e-6)

if sol.converged:
    # 打印找到的 f''(0) 初值
    f_double_prime_0 = sol.root
    print(f"Converged: f''(0) = {f_double_prime_0}")

    # 使用找到的初值进行整体积分计算
    y0 = [0, 0, f_double_prime_0]
    x_span = [0, 10]
    x_eval = np.linspace(0, 10, 500)
    solution = solve_ivp(blasius_ode, x_span, y0, t_eval=x_eval, method='RK45')

    # 获取解的各个分量
    f = solution.y[0]
    f_prime = solution.y[1]

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(solution.t, f, label='f(x)', color='b', linestyle='--')
    plt.plot(solution.t, f_prime, label="f'(x)", color='r', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Solution')
    plt.title('Blasius Equation Solution using Shooting Method')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Root finding did not converge.")

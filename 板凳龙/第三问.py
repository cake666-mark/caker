import numpy as np

# 参数
target_radius = 4.5  # 调头空间的半径（单位：米）
initial_circle = 16  # 龙头初始位置的圈数
initial_theta = initial_circle * 2 * np.pi  # 初始角度
final_theta = None  # 最终角度，待求解

# 计算螺距P的函数
def compute_min_pitch(target_radius, initial_theta):
    # 目标极径为4.5米，解出最终的角度
    final_theta = initial_theta
    return (target_radius * 2 * np.pi) / final_theta

# 计算最小螺距P
min_pitch = compute_min_pitch(target_radius, initial_theta)
print(f"最小螺距为：{min_pitch:.6f} 米")
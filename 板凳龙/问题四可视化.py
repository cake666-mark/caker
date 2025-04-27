import numpy as np
import matplotlib.pyplot as plt

# 参数
v_head = 1.0  # 龙头速度 1 m/s
P_in = 1.7  # 盘入螺线螺距 (m)
turn_radius1 = 4.5  # 第一段圆弧的半径 (m)
turn_radius2 = turn_radius1 / 2  # 第二段圆弧的半径是第一段的二分之一
time_step = 1  # 时间步长 1 秒
start_time = -100  # 起始时间 (秒)
end_time = 100  # 结束时间 (秒)
total_time = end_time - start_time

# 时间轴
times = np.arange(start_time, end_time + time_step, time_step)

# 圆心坐标（假设调头区域中心在 (0,0)）
center1 = (-turn_radius1, 0)
center2 = (turn_radius2, 0)

# 初始化位置和速度数组
positions = []
velocities = []

# 计算路径（S形曲线）
for t in times:
    if t < 0:  # 第一个圆弧
        theta = v_head * t / turn_radius1  # 角度
        x = center1[0] + turn_radius1 * np.cos(theta)
        y = turn_radius1 * np.sin(theta)
    else:  # 第二个圆弧
        theta = v_head * t / turn_radius2
        x = center2[0] + turn_radius2 * np.cos(theta)
        y = turn_radius2 * np.sin(theta)

    # 计算速度（速度保持 1 m/s 不变）
    speed = v_head

    positions.append((x, y))
    velocities.append(speed)

# 转换为 NumPy 数组以便绘图
positions = np.array(positions)

# 修正关键节点时间步
key_points = [0, 1, 51, 101, 151, min(201, len(times) - 1)]  # 确保关键节点在时间范围内

# 创建可视化
plt.figure(figsize=(10, 8))

# 绘制S形路径
plt.plot(positions[:, 0], positions[:, 1], label="S-curve Path")

# 标记龙头及其后面的关键节点位置
for i in key_points:
    plt.plot(positions[i, 0], positions[i, 1], 'ro')
    plt.text(positions[i, 0], positions[i, 1], f"t={times[i]}s", fontsize=12)

# 设置图表标题和标签
plt.title("Dragon Dance Turning Path Visualization (S-shaped Curve)")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.grid(True)
plt.legend()

# 展示图表
plt.show()
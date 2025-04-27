import numpy as np
import matplotlib.pyplot as plt

# 螺距和圈数
pitch = 55  # 螺距为55cm
num_turns = 16  # 螺线有16圈
total_time = 300  # 总时间300秒
initial_speed = 1  # 初始速度1m/s
n_segments=223  # 螺旋线上总共1000个点

# 生成角度theta的范围，反转theta以便龙从外到内移动
theta = np.linspace(0, num_turns * 2 * np.pi, n_segments)[::-1]  # 反转顺序

# 计算螺旋线的半径，随着角度减小，半径也会减小
r = (pitch / (2 * np.pi)) * theta

# 将极坐标转换为直角坐标
x = r * np.cos(theta)
y = r * np.sin(theta)

# 计算总螺旋线长度
distance = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
distance = np.insert(distance, 0, 0)  # 插入初始距离

# 根据龙头初始速度计算每秒移动的距离
speed = initial_speed
total_distance = speed * total_time  # 300秒后移动的总距离

# 找到龙头在300秒后的位置
index = np.searchsorted(distance, total_distance)

# 创建图形
plt.figure(figsize=(6, 6))

# 绘制完整螺旋线
plt.plot(x, y, color='gray', alpha=0.5, label=f'Spiral with {pitch}cm pitch, {num_turns} turns')

# 绘制龙身的线段（蓝色）
plt.plot(x[:index], y[:index], color='blue', label='Dragon Body')

# 绘制龙头（红色）
plt.plot(x[index], y[index], 'ro', markersize=10, label='Dragon Head')

# 设置标题和坐标轴标签
plt.title('Dragon Position after 300 seconds on the Spiral (Moving from Outside to Inside)')
plt.xlabel('X axis (cm)')
plt.ylabel('Y axis (cm)')

# 显示网格和图例
plt.grid(True)
plt.legend()

# 显示图形
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 创建绘图窗口
fig, ax = plt.subplots(figsize=(6, 6))

# 设置黄色圆形区域的参数
yellow_radius = 1.5  # 黄色区域的半径
yellow_circle = Circle((0, 0), yellow_radius, color='yellow', ec='black', linestyle='--')
ax.add_patch(yellow_circle)  # 绘制黄色圆

# 设置螺旋线的参数
theta = np.linspace(0, 10 * np.pi, 1000)  # 5圈螺旋
a = yellow_radius  # 从黄色圆的边缘开始
r = a + 0.1 * theta  # 螺旋线的半径

# 将极坐标转换为笛卡尔坐标
x = r * np.cos(theta)
y = r * np.sin(theta)

# 绘制螺旋线
ax.plot(x, y, 'b', lw=2)  # 蓝色的螺旋线

# 绘制黄色区域内部的倒S型路径
s_x = np.linspace(-yellow_radius, yellow_radius, 500)  # X坐标范围在黄色圆内
s_y = -0.5 * np.sin(s_x * np.pi / yellow_radius)  # Y坐标生成倒S型曲线
ax.plot(s_x, s_y, 'r', lw=2)  # 红色的倒S型路径

# 设置坐标轴比例和边界
ax.set_aspect('equal')
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])

# 移除坐标轴
ax.axis('off')

# 显示图形
plt.show()
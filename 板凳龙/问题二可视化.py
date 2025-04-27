import numpy as np
import matplotlib.pyplot as plt

# Known conditions (用中文标注参数)
pitch = 0.55  # 螺距，单位：米
min_safe_distance = 0.3  # 板凳之间的最小安全距离，单位：米
head_speed = 1.0  # 龙头的速度，单位：米/秒
initial_turns = 16  # 初始螺旋的圈数

# Time settings (时间设置)
time_step = 1  # 时间步长为1秒
max_time = 300  # 最大模拟时间，单位：秒


# Function to calculate the radius of the spiral (计算螺线半径)
def calculate_spiral_radius(theta, pitch):
    # 根据极角theta和螺距pitch计算螺线半径
    return pitch * theta / (2 * np.pi)


# Function to calculate the distance between two points (计算两点间的距离)
def calculate_distance_between_benches(theta1, theta2, pitch):
    radius1 = calculate_spiral_radius(theta1, pitch)
    radius2 = calculate_spiral_radius(theta2, pitch)
    return abs(radius1 - radius2)


# Initialize variables (初始化变量)
head_theta = initial_turns * 2 * np.pi  # 初始龙头的极角
time = 0
theta_history = []
time_history = []

# Simulation loop (模拟过程)
while time <= max_time:
    # Record current angle and time for the dragon head (记录当前龙头的角度和时间)
    theta_history.append(head_theta)
    time_history.append(time)

    # Calculate the distance between the dragon head and the 1st bench (计算龙头和第1节龙身之间的距离)
    body_theta = head_theta - 2 * np.pi  # 相邻板凳的极角差为一圈
    bench_distance = calculate_distance_between_benches(head_theta, body_theta, pitch)

    # Check for collision (判断是否发生碰撞)
    if bench_distance < min_safe_distance:
        print(f"The distance between benches is {bench_distance:.6f} meters, collision occurs.")
        print(f"Time of termination: {time} seconds")
        break

    # Update head position and time (更新龙头的位置和时间)
    head_theta += head_speed / calculate_spiral_radius(head_theta, pitch)  # 根据速度更新角度
    time += time_step


# Function to generate spiral trajectory data (获取螺线轨迹数据)
def plot_spiral(turns, pitch):
    theta = np.linspace(0, turns * 2 * np.pi, 1000)
    radius = pitch * theta / (2 * np.pi)
    return theta, radius


# Convert polar coordinates to Cartesian coordinates (极坐标转换为直角坐标)
def polar_to_cartesian(theta, radius):
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


# Generate the spiral and dragon head trajectory (绘制16圈的螺线和龙头实际路径)
theta_16_turns, radius_16_turns = plot_spiral(initial_turns, pitch)
theta_history_array = np.array(theta_history)
radius_history = calculate_spiral_radius(theta_history_array, pitch)

# Convert spiral and actual path to Cartesian coordinates (将螺线和实际路径转换为直角坐标)
x_16_turns, y_16_turns = polar_to_cartesian(theta_16_turns, radius_16_turns)
x_history, y_history = polar_to_cartesian(theta_history_array, radius_history)

# Plot the spiral and dragon head path (绘制二维坐标系图像)
plt.figure(figsize=(8, 8))

# Plot 16-turn spiral, in blue (绘制16圈螺线，颜色设置为蓝色)
plt.plot(x_16_turns, y_16_turns, label="16-turn spiral", color="blue", linestyle='--')

# Plot dragon head path, in orange (绘制龙头实际轨迹，颜色设置为橙色)
plt.plot(x_history, y_history, label="Dragon head path", color="orange")

# Add legend, title, and labels (添加图例、标题和标签)
plt.title("Dragon head path along 16-turn spiral", fontsize=14)
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.legend()

# Ensure equal scaling (保证等比例显示)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

# Show the plot (显示图像)
plt.show()
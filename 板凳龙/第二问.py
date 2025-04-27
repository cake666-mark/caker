import numpy as np

# 参数
P = 0.70  # 螺距（单位：米），尝试增加值
v_head = 1.2  # 龙头速度（单位：米/秒），尝试增加值
initial_theta = 16 * 2 * np.pi  # 初始角度
r_head_0 = P / (2 * np.pi) * initial_theta  # 初始半径
num_segments = 223  # 总节数
segment_length = 2.2  # 每节板凳的长度（单位：米）

# 时间步长
delta_t = 1  # 时间间隔0.1秒
t_max = 5000  # 增加最大计算时间限制为5000秒
times = np.arange(0, t_max, delta_t)


# 函数：计算给定角度的螺旋半径
def radius(theta):
    return P / (2 * np.pi) * theta


# 函数：极坐标转换为笛卡尔坐标
def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


# 函数：计算两点之间的距离
def distance_between_points(r1, theta1, r2, theta2):
    x1, y1 = polar_to_cartesian(r1, theta1)
    x2, y2 = polar_to_cartesian(r2, theta2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 函数：检查是否发生碰撞
def check_collision(time):
    theta_head = initial_theta + v_head * time / r_head_0
    r_head = radius(theta_head)

    previous_theta = theta_head
    previous_r = r_head

    for i in range(1, num_segments):
        current_theta = theta_head - (i * segment_length / r_head_0)
        current_r = radius(current_theta)

        distance = distance_between_points(previous_r, previous_theta, current_r, current_theta)

        if distance < segment_length:
            return True, time

        previous_theta = current_theta
        previous_r = current_r

    return False, None


# 找到碰撞时刻
collision_time = None
for t in times:
    collision, time = check_collision(t)
    if collision:
        collision_time = time
        break

# 计算碰撞时刻的各部分位置和速度
if collision_time is not None:
    print(f"碰撞发生在 {collision_time:.2f} 秒")

    theta_head = initial_theta + v_head * collision_time / r_head_0
    r_head = radius(theta_head)
    x_head, y_head = polar_to_cartesian(r_head, theta_head)
    v_head_x = -v_head * np.sin(theta_head)
    v_head_y = v_head * np.cos(theta_head)

    print(f"龙头位置：({x_head:.6f}, {y_head:.6f})，速度：({v_head_x:.6f}, {v_head_y:.6f})")

    for i in [1, 51, 101, 151, 201]:
        segment_theta = initial_theta + v_head * collision_time / r_head_0 - (i * segment_length / r_head_0)
        r_segment = radius(segment_theta)
        x_segment, y_segment = polar_to_cartesian(r_segment, segment_theta)
        v_segment_x = -v_head * np.sin(segment_theta)
        v_segment_y = v_head * np.cos(segment_theta)
        print(f"龙身第{i}节位置：({x_segment:.6f}, {y_segment:.6f})，速度：({v_segment_x:.6f}, {v_segment_y:.6f})")

    theta_tail = initial_theta + v_head * collision_time / r_head_0 - (num_segments * segment_length / r_head_0)
    r_tail = radius(theta_tail)
    x_tail, y_tail = polar_to_cartesian(r_tail, theta_tail)
    v_tail_x = -v_head * np.sin(theta_tail)
    v_tail_y = v_head * np.cos(theta_tail)

    print(f"龙尾位置：({x_tail:.6f}, {y_tail:.6f})，速度：({v_tail_x:.6f}, {v_tail_y:.6f})")
else:
    print("未发生碰撞")
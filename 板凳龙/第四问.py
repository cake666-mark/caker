import numpy as np

# 参数
v_head = 1  # 龙头速度（1 m/s）
P_in = 1.7  # 盘入螺线螺距（米）
P_out = 1.7  # 盘出螺线螺距（米）
R1 = 4.0  # 前段圆弧半径（可调，单位：米）
R2 = 2 * R1  # 后段圆弧半径
t_turn_start = 0  # 调头开始时间
times = np.arange(-100, 101, 1)  # 时间范围从-100秒到100秒，每秒记录

# 函数：计算螺线位置（极坐标）
def spiral_position(P, theta):
    r = P / (2 * np.pi) * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# 函数：计算S形调头曲线位置
def s_curve_position(t, R1, R2, v_head):
    if t < t_turn_start:
        # 盘入螺线位置
        theta = v_head * t / (P_in / (2 * np.pi))
        return spiral_position(P_in, theta)
    elif t >= t_turn_start:
        # 假设调头区为圆弧，简化计算，返回圆弧上的位置
        angle_R1 = v_head * t / R1  # 前段圆弧的角度
        x_R1 = R1 * np.cos(angle_R1)
        y_R1 = R1 * np.sin(angle_R1)
        return x_R1, y_R1

# 函数：计算速度
def speed(t, v_head):
    if t < t_turn_start:
        return v_head, 0  # 盘入螺线，速度沿着螺线前进
    else:
        return v_head, v_head  # 调头区，假设速度为常数

# 计算各个时刻龙头及各节龙身位置和速度
for t in [-100, -50, 0, 50, 100]:
    # 计算龙头前把手位置和速度
    x_head, y_head = s_curve_position(t, R1, R2, v_head)
    v_head_x, v_head_y = speed(t, v_head)
    print(f"时间 {t} 秒，龙头前把手位置：({x_head:.6f}, {y_head:.6f})，速度：({v_head_x:.6f}, {v_head_y:.6f})")

    # 计算龙身第1、51、101、151、201节位置和速度
    for i in [1, 51, 101, 151, 201]:
        x_segment, y_segment = s_curve_position(t - i * 2.2 / v_head, R1, R2, v_head)
        v_segment_x, v_segment_y = speed(t - i * 2.2 / v_head, v_head)
        print(f"时间 {t} 秒，龙身第{i}节位置：({x_segment:.6f}, {y_segment:.6f})，速度：({v_segment_x:.6f}, {v_segment_y:.6f})")

    # 计算龙尾后把手位置和速度（第223节）
    x_tail, y_tail = s_curve_position(t - 223 * 2.2 / v_head, R1, R2, v_head)
    v_tail_x, v_tail_y = speed(t - 223 * 2.2 / v_head, v_head)
    print(f"时间 {t} 秒，龙尾后把手位置：({x_tail:.6f}, {y_tail:.6f})，速度：({v_tail_x:.6f}, {v_tail_y:.6f})")
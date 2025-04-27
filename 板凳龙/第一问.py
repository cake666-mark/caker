import numpy as np

# 定义常量
pitch = 55  # 螺距，cm
v_head = 100  # 龙头前把手速度，cm/s
r_initial = 16 * pitch  # 初始半径，第16圈，cm
section_length = 220  # 每节龙身的长度，cm
time_points = [0, 60, 120, 180, 240, 300]  # 关心的时间点


# 计算给定时间时龙头的位置
def head_position(t):
    # 沿着螺旋线的路径长度 s
    s = v_head * t
    # 计算角度 theta
    theta = s / pitch
    # 计算半径 r
    r = r_initial - pitch * theta / (2 * np.pi)
    # 计算 x, y 位置（极坐标转笛卡尔坐标）
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, r


# 计算每节龙身/龙尾的速度
def calculate_section_speed(head_speed, r_head, r_body):
    # 根据半径变化调整速度
    if r_body != 0:
        return head_speed * (r_body / r_head)
    else:
        return 0


# 计算每个时间点的龙头、龙身、龙尾的位置和速度
def calculate_positions_speeds():
    positions_speeds = {}

    for t in time_points:
        positions_speeds[t] = {}

        # 计算龙头前把手的坐标和速度
        x_head, y_head, r_head = head_position(t)
        positions_speeds[t]["head"] = {
            "position": (x_head / 100, y_head / 100),  # 转换为米
            "speed": v_head / 100  # 转换为m/s
        }

        # 计算龙身第 1, 51, 101, 151, 201 节前把手的坐标和速度
        for i in [1, 51, 101, 151, 201]:
            if t - i * section_length / v_head >= 0:
                x_body, y_body, r_body = head_position(t - i * section_length / v_head)
                speed_body = calculate_section_speed(v_head, r_head, r_body)
            else:
                x_body, y_body, r_body = head_position(0)
                speed_body = 0
            positions_speeds[t][f"body_{i}"] = {
                "position": (x_body / 100, y_body / 100),  # 转换为米
                "speed": speed_body / 100  # 转换为m/s
            }

        # 计算龙尾后把手的坐标和速度（第 221 节）
        if t - 221 * section_length / v_head >= 0:
            x_tail, y_tail, r_tail = head_position(t - 221 * section_length / v_head)
            speed_tail = calculate_section_speed(v_head, r_head, r_tail)
        else:
            x_tail, y_tail, r_tail = head_position(0)
            speed_tail = 0
        positions_speeds[t]["tail"] = {
            "position": (x_tail / 100, y_tail / 100),  # 转换为米
            "speed": speed_tail / 100  # 转换为m/s
        }

    return positions_speeds


# 计算结果
positions_speeds = calculate_positions_speeds()

# 打印结果
for t in time_points:
    print(f"时间: {t} 秒")
    print(
        f"龙头前把手: 位置 = {positions_speeds[t]['head']['position']}, 速度 = {positions_speeds[t]['head']['speed']} m/s")
    for i in [1, 51, 101, 151, 201]:
        print(
            f"第 {i} 节龙身前把手: 位置 = {positions_speeds[t][f'body_{i}']['position']}, 速度 = {positions_speeds[t][f'body_{i}']['speed']} m/s")
    print(
        f"龙尾后把手: 位置 = {positions_speeds[t]['tail']['position']}, 速度 = {positions_speeds[t]['tail']['speed']} m/s")
    print("-" * 50)

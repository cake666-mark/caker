import numpy as np

# 参数
v_max_segment = 2.0  # 板凳把手的最大速度 (m/s)
P_in = 1.7  # 盘入螺线螺距 (m)
R1 = 4.0  # 调头前段圆弧半径 (m)
R2 = 2 * R1  # 调头后段圆弧半径 (m)

# 计算螺线段的曲率和速度
def spiral_speed(v_head, P, theta):
    # 计算螺线段的曲率
    r = P / (2 * np.pi) * theta
    if r == 0:
        return v_head  # 初始位置速度直接等于龙头速度
    curvature = 1 / r
    # 速度增长控制在合理范围
    v_segment = v_head * (1 + 0.1 * curvature)  # 放慢曲率对速度的增长影响
    return v_segment

# 计算调头段的速度
def turn_speed(v_head, R):
    # 调头段速度与半径成反比，加入限制
    v_segment = v_head * (1 + 0.1 * (R1 / R))  # 控制速度增量
    return v_segment

# 函数：确定龙头最大速度
def compute_max_head_speed():
    # 遍历不同路径段，找到最大速度不超过2 m/s的龙头速度
    v_head = 0.1  # 初始龙头速度
    while True:
        # 计算螺线段速度
        for theta in np.arange(0.1, 2 * np.pi * 10, 0.1):  # 假设螺线走了10圈
            v_segment = spiral_speed(v_head, P_in, theta)
            if v_segment > v_max_segment:
                return v_head

        # 计算调头段速度
        v_turn1 = turn_speed(v_head, R1)  # 前段圆弧
        v_turn2 = turn_speed(v_head, R2)  # 后段圆弧
        if max(v_turn1, v_turn2) > v_max_segment:
            return v_head

        # 增加龙头速度
        v_head += 0.05  # 增加步长进行逼近（从0.01调高到0.05）

# 计算龙头最大速度
v_head_max = compute_max_head_speed()
print(f"龙头的最大行进速度为：{v_head_max:.6f} m/s")
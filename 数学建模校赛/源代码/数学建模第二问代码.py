import numpy as np

# 常量：经纬度（单位：度）
latitude = 30 + 35 / 60  # 北纬 30°35′
longitude = 114 + 19 / 60  # 东经 114°19′

# 5月23日的近似赤纬角
declination = 20

# 计算角度的函数
def calculate_angles(time_str, latitude, declination):
    # 将输入的时间字符串（HH:MM）转换为小时数（浮点数）
    hour = int(time_str.split(':')[0])
    minute = int(time_str.split(':')[1])
    local_time = hour + minute / 60
    hour_angle = 15 * (local_time - 12)  # 计算时角（单位：度）

    # 将经度、赤纬角和时角转换为弧度
    phi = np.radians(latitude)
    delta = np.radians(declination)
    h = np.radians(hour_angle)

    # 计算太阳高度角（单位：度）
    sin_alpha = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(h)
    altitude_angle = np.degrees(np.arcsin(sin_alpha))

    # 计算太阳方位角（单位：度）
    if np.abs(sin_alpha) <= 1:
        cos_A = (np.sin(delta) * np.cos(phi) - np.cos(delta) * np.sin(phi) * np.cos(h)) / np.cos(np.radians(altitude_angle))
        cos_A = np.clip(cos_A, -1, 1)  # 限制在 [-1, 1] 范围内
        azimuth_angle = np.degrees(np.arccos(cos_A))
    else:
        azimuth_angle = np.nan  # 如果计算不可行，则返回 NaN

    return altitude_angle, azimuth_angle

# 交互式输入部分
if __name__ == "__main__":
    while True:
        try:
            time_input = input("请输入时间 (HH:MM)，输入 q 退出：")
            if time_input.lower() == 'q':
                break

            altitude, azimuth = calculate_angles(time_input, latitude, declination)
            if np.isnan(azimuth):
                print(f"在 {time_input} 时，无法计算方位角。")
            else:
                print(f"在 {time_input} 时，太阳的高度角为: {altitude:.2f}°，方位角为: {azimuth:.2f}°")

        except ValueError:
            print("时间格式错误，请输入正确的时间格式 (HH:MM)。")

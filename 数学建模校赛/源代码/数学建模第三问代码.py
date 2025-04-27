import datetime
import math


def solar_position(latitude, longitude, date_time):
    # 计算太阳位置，使用合适的库或算法
    # 示例：使用 PyEphem 库或 SolarPosition 类

    # 替换为实际的计算方法
    solar_altitude = calculate_solar_altitude(latitude, longitude, date_time)
    solar_azimuth = calculate_solar_azimuth(latitude, longitude, date_time)

    return solar_altitude, solar_azimuth


def calculate_solar_altitude(latitude, longitude, date_time):
    # 将纬度和经度转换为弧度
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # 计算一年中的日数
    day_of_year = date_time.timetuple().tm_yday

    # 计算太阳赤纬角（delta）
    delta = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))

    # 计算时角（H）
    hour_angle = (date_time.hour - 12) * 15 + (date_time.minute / 60) * 15 + longitude

    # 计算太阳高度角（h）
    h = math.degrees(math.asin(math.sin(lat_rad) * math.sin(math.radians(delta)) +
                               math.cos(lat_rad) * math.cos(math.radians(delta)) *
                               math.cos(math.radians(hour_angle))))

    return h


def calculate_solar_azimuth(latitude, longitude, date_time):
    # 将纬度和经度转换为弧度
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # 计算一年中的日数
    day_of_year = date_time.timetuple().tm_yday

    # 计算太阳赤纬角（delta）
    delta = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))

    # 计算时角（H）
    hour_angle = (date_time.hour - 12) * 15 + (date_time.minute / 60) * 15 + longitude

    # 计算太阳方位角（A）
    A = math.degrees(math.atan2(math.sin(math.radians(hour_angle)),
                                math.cos(math.radians(hour_angle)) * math.sin(lat_rad) -
                                math.tan(math.radians(delta)) * math.cos(lat_rad)))

    # 调整方位角使其在 0 到 360 度之间
    azimuth = (A + 360) % 360

    return azimuth


def adjust_mirror(latitude, longitude, start_time, end_time, time_interval):
    adjustment_data = []
    current_time = start_time

    while current_time <= end_time:
        # 获取当前太阳位置
        solar_altitude, solar_azimuth = solar_position(latitude, longitude, current_time)

        # 计算凹面镜调整角度
        azimuth_adjustment = solar_azimuth - longitude
        elevation_adjustment = 90 - solar_altitude

        # 添加调整数据到列表中
        adjustment_data.append((current_time, azimuth_adjustment, elevation_adjustment))

        # 增加时间间隔
        current_time += datetime.timedelta(minutes=time_interval)

    return adjustment_data


# 示例用法:
city_latitude = 30.5833  # 北纬 30°35′ 转换为小数形式
city_longitude = 114.3167  # 东经 114°19′ 转换为小数形式
start_time = datetime.datetime(2023, 5, 23, 8, 0, 0)  # 开始时间
end_time = datetime.datetime(2023, 5, 23, 18, 0, 0)  # 结束时间
time_interval = 15  # 时间间隔为 15 分钟

adjustment_data = adjust_mirror(city_latitude, city_longitude, start_time, end_time, time_interval)

# 输出调整数据
for current_time, azimuth_adjustment, elevation_adjustment in adjustment_data:
    print(
        f"时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}, 方位角调整: {azimuth_adjustment:.2f} 度, 高度角调整: {elevation_adjustment:.2f} 度")

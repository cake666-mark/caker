import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from itertools import permutations
from scipy.optimize import linear_sum_assignment
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# 1. 定义常量和参数
VEHICLE_SPEED = 25  # 检修车速度 (km/h)
MAX_BIKES_PER_VEHICLE = 20  # 每辆检修车最大运载量
FAULT_RATE = 0.06  # 故障率 6%
HANDLING_TIME_PER_BIKE = 1  # 每辆自行车的搬运时间 (分钟)

# 2. 创建模拟数据
# 2.1 定义各停车点及检修处的位置（假设基于校园地图坐标）
locations = {
    '检修处': (0, 0),  # 检修处在坐标原点（校园东北角）
    '东门': (0.2, -0.3),
    '南门': (0.3, -1.2),
    '北门': (-0.3, 0.1),
    '一食堂': (0.5, -0.5),
    '二食堂': (0.8, -0.6),
    '三食堂': (0.7, -0.2),
    '梅苑1栋': (-0.2, -0.4),
    '菊苑1栋': (-0.4, -0.7),
    '教学2楼': (0.6, -0.8),
    '教学4楼': (0.9, -0.9),
    '计算机学院': (0.4, -1.0),
    '工程中心': (0.1, -0.8),
    '网球场': (-0.5, -0.5),
    '体育馆': (-0.7, -0.3),
    '校医院': (-0.4, -0.1)
}

# 假设的优化后的单车分布（根据问题1的结果）
# 这里使用问题1中计算的平均值
bike_distribution = {
    '东门': 46,
    '南门': 60,
    '北门': 54,
    '一食堂': 59,
    '二食堂': 73,
    '三食堂': 60,
    '梅苑1栋': 68,
    '菊苑1栋': 75,
    '教学2楼': 58,
    '教学4楼': 41,
    '计算机学院': 35,
    '工程中心': 15,
    '网球场': 14,
    '体育馆': 10,
    '校医院': 12
}


# 3. 计算距离矩阵
def calculate_distances():
    """计算各停车点到检修处的距离"""
    distances = {}
    for location, coords in locations.items():
        if location != '检修处':
            # 计算欧几里得距离
            dx = coords[0] - locations['检修处'][0]
            dy = coords[1] - locations['检修处'][1]
            distances[location] = np.sqrt(dx ** 2 + dy ** 2)
    return distances


# 4. 计算故障车辆数量
def calculate_faulty_bikes():
    """计算各停车点的故障车辆数量"""
    faulty_bikes = {}
    for location, bikes in bike_distribution.items():
        faulty_bikes[location] = round(bikes * FAULT_RATE)
    return faulty_bikes


# 5. 计算每个停车点到检修处的往返时间
def calculate_round_trip_times(distances):
    """计算往返时间（小时）"""
    round_trip_times = {}
    for location, distance in distances.items():
        # 往返距离除以速度（单位：小时）
        round_trip_times[location] = (2 * distance) / VEHICLE_SPEED
    return round_trip_times


# 6. 计算从各停车点运回故障车辆所需的总时间
def calculate_total_times(round_trip_times, faulty_bikes):
    """计算从各停车点运回故障车辆所需的总时间（分钟）"""
    total_times = {}
    for location in round_trip_times:
        # 计算往返时间（分钟）
        travel_time = round_trip_times[location] * 60
        # 查找和搬运故障车辆的时间
        handling_time = faulty_bikes[location] * HANDLING_TIME_PER_BIKE
        # 计算总时间
        total_times[location] = travel_time + handling_time
    return total_times


# 7. 计算需要的行程次数
def calculate_trips_needed(faulty_bikes):
    """计算从各停车点运回故障车辆需要的行程次数"""
    trips_needed = {}
    for location, bikes in faulty_bikes.items():
        trips_needed[location] = np.ceil(bikes / MAX_BIKES_PER_VEHICLE)
    return trips_needed


# 8. 贪心算法寻找最优检修路径
def greedy_repair_route(round_trip_times, faulty_bikes):
    """使用贪心算法寻找最优检修路径"""
    # 计算每个位置的单位时间内可修复的车辆数
    efficiency = {}
    for location in round_trip_times:
        # 效率 = 可修复的车辆数 / 往返时间
        if round_trip_times[location] > 0:
            efficiency[location] = min(faulty_bikes[location], MAX_BIKES_PER_VEHICLE) / (
                        round_trip_times[location] * 60)
        else:
            efficiency[location] = float('inf')  # 避免除以零

    # 按效率从高到低排序
    sorted_locations = sorted(efficiency.keys(), key=lambda x: efficiency[x], reverse=True)

    return sorted_locations


# 9. 实现最近邻算法（TSP）来优化多次行程的路径
def nearest_neighbor_tsp(locations_to_visit, starting_point='检修处'):
    """使用最近邻算法求解TSP问题"""
    # 创建完整的距离矩阵（包括所有地点之间的距离）
    all_locations = list(locations_to_visit) + [starting_point]
    n = len(all_locations)
    distance_matrix = np.zeros((n, n))

    # 填充距离矩阵
    for i in range(n):
        for j in range(n):
            if i != j:
                loc1 = locations[all_locations[i]]
                loc2 = locations[all_locations[j]]
                distance_matrix[i][j] = np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    # 使用最近邻算法
    start_idx = all_locations.index(starting_point)
    unvisited = set(range(n))
    unvisited.remove(start_idx)

    path = [start_idx]
    while unvisited:
        current = path[-1]
        # 找到距离当前位置最近的未访问点
        next_point = min(unvisited, key=lambda x: distance_matrix[current][x])
        path.append(next_point)
        unvisited.remove(next_point)

    # 添加回到起点的路径
    path.append(start_idx)

    # 转换为地点名称
    route = [all_locations[i] for i in path]
    return route


# 10. 计算优化后的巡检时间
def calculate_optimized_repair_time(route, faulty_bikes, locations_dict):
    """计算优化后的巡检路径所需的总时间"""
    total_distance = 0
    total_handling_time = 0

    # 计算总距离
    for i in range(len(route) - 1):
        loc1 = locations_dict[route[i]]
        loc2 = locations_dict[route[i + 1]]
        distance = np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
        total_distance += distance

    # 计算总处理时间
    for location in route:
        if location in faulty_bikes:
            total_handling_time += faulty_bikes[location] * HANDLING_TIME_PER_BIKE

    # 计算总时间（分钟）
    travel_time = (total_distance / VEHICLE_SPEED) * 60
    total_time = travel_time + total_handling_time

    return total_time, travel_time, total_handling_time


# 11. 贪心算法结合车辆容量约束的优化模型
def optimize_repair_schedule(faulty_bikes, round_trip_times):
    """优化检修车辆的调度计划"""
    locations_list = list(faulty_bikes.keys())
    remaining_bikes = faulty_bikes.copy()
    schedule = []

    total_trips = sum(np.ceil(bikes / MAX_BIKES_PER_VEHICLE) for bikes in faulty_bikes.values())
    print(f"总计需要 {total_trips} 次行程来运回所有故障车辆")

    while any(bikes > 0 for bikes in remaining_bikes.values()):
        # 初始化当前行程
        current_trip = {'route': [], 'bikes_collected': {}}
        remaining_capacity = MAX_BIKES_PER_VEHICLE

        # 基于效率选择停车点
        for location in greedy_repair_route(round_trip_times, remaining_bikes):
            if remaining_bikes[location] > 0 and remaining_capacity > 0:
                # 确定可以收集的单车数量
                bikes_to_collect = min(remaining_bikes[location], remaining_capacity)

                # 更新容量和剩余故障车辆
                remaining_capacity -= bikes_to_collect
                remaining_bikes[location] -= bikes_to_collect

                # 记录路径和收集数量
                current_trip['route'].append(location)
                current_trip['bikes_collected'][location] = bikes_to_collect

                if remaining_capacity == 0:
                    break

        # 使用TSP优化当前行程的路径
        if current_trip['route']:
            optimized_route = nearest_neighbor_tsp(current_trip['route'])
            current_trip['optimized_route'] = optimized_route

            # 计算这次行程的时间
            trip_time, travel_time, handling_time = calculate_optimized_repair_time(
                optimized_route, current_trip['bikes_collected'], locations)

            current_trip['total_time'] = trip_time
            current_trip['travel_time'] = travel_time
            current_trip['handling_time'] = handling_time

            schedule.append(current_trip)

    return schedule


# 12. 计算最佳巡检时间
def find_best_inspection_time(schedule, bike_usage_pattern):
    """找出最佳的巡检时间，考虑自行车使用模式"""
    # 这里使用简化的自行车使用模式，实际应基于问题1的分析
    # 假设使用模式为一天中不同时间段的使用率（0-1之间）
    # 时间越低，表示空闲车辆越多，故障检修越合适

    # 计算每个时间段完成所有检修所需的总时间
    total_repair_time = sum(trip['total_time'] for trip in schedule)

    # 将总时间转换为小时
    total_hours = total_repair_time / 60

    # 找出使用率最低的连续时间段，时间长度为total_hours
    best_start_time = None
    lowest_usage = float('inf')

    for start_hour in range(24):
        end_hour = (start_hour + int(np.ceil(total_hours))) % 24

        # 计算这个时间段内的平均使用率
        if end_hour > start_hour:
            avg_usage = np.mean([bike_usage_pattern[h % 24] for h in range(start_hour, end_hour)])
        else:  # 跨越午夜
            hours = list(range(start_hour, 24)) + list(range(0, end_hour))
            avg_usage = np.mean([bike_usage_pattern[h % 24] for h in hours])

        if avg_usage < lowest_usage:
            lowest_usage = avg_usage
            best_start_time = start_hour

    return best_start_time, (best_start_time + int(np.ceil(total_hours))) % 24


# 13. 可视化路径
def visualize_repair_route(schedule, locations_dict):
    """可视化检修路径"""
    plt.figure(figsize=(12, 10))

    # 绘制所有地点
    for location, (x, y) in locations_dict.items():
        if location == '检修处':
            plt.scatter(x, y, s=200, color='red', marker='*', label='检修处')
        else:
            plt.scatter(x, y, s=100, color='blue')
            plt.annotate(location, (x, y), fontsize=8)

    # 绘制路径
    colors = plt.cm.jet(np.linspace(0, 1, len(schedule)))

    for i, trip in enumerate(schedule):
        route = trip['optimized_route']
        for j in range(len(route) - 1):
            x1, y1 = locations_dict[route[j]]
            x2, y2 = locations_dict[route[j + 1]]
            plt.plot([x1, x2], [y1, y2], '-', color=colors[i], alpha=0.7,
                     linewidth=2)

        # 在第一个点添加行程编号
        x, y = locations_dict[route[1]]  # 第一个实际停车点（不是检修处）
        plt.annotate(f"行程{i + 1}", (x, y), xytext=(5, 5),
                     textcoords='offset points', color=colors[i], fontweight='bold')

    plt.title('共享单车故障检修最优路径')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('repair_route.png')
    plt.show()


# 14. 主函数
def main():
    # 1. 计算各停车点到检修处的距离
    distances = calculate_distances()
    print("各停车点到检修处的距离（千米）:")
    for location, distance in distances.items():
        print(f"{location}: {distance:.2f} km")

    # 2. 计算各停车点的故障车辆数量
    faulty_bikes = calculate_faulty_bikes()
    print("\n各停车点的故障车辆数量:")
    for location, bikes in faulty_bikes.items():
        print(f"{location}: {bikes} 辆")

    total_faulty_bikes = sum(faulty_bikes.values())
    print(f"\n总故障车辆数量: {total_faulty_bikes} 辆")

    # 3. 计算往返时间
    round_trip_times = calculate_round_trip_times(distances)
    print("\n各停车点到检修处的往返时间（小时）:")
    for location, time in round_trip_times.items():
        print(f"{location}: {time:.2f} 小时")

    # 4. 计算所需行程次数
    trips_needed = calculate_trips_needed(faulty_bikes)
    print("\n各停车点需要的行程次数:")
    for location, trips in trips_needed.items():
        print(f"{location}: {int(trips)} 次行程")

    # 5. 优化检修调度计划
    schedule = optimize_repair_schedule(faulty_bikes, round_trip_times)

    # 输出优化后的调度计划
    print("\n优化后的检修调度计划:")
    total_time = 0
    for i, trip in enumerate(schedule):
        print(f"\n行程 {i + 1}:")
        print(f"路径: {' -> '.join(trip['optimized_route'])}")
        print(f"收集的故障车辆: {trip['bikes_collected']}")
        print(f"总时间: {trip['total_time']:.2f} 分钟")
        print(f"行驶时间: {trip['travel_time']:.2f} 分钟")
        print(f"处理时间: {trip['handling_time']:.2f} 分钟")
        total_time += trip['total_time']

    print(f"\n完成所有检修所需的总时间: {total_time:.2f} 分钟 ({total_time / 60:.2f} 小时)")

    # 6. 计算最佳巡检时间
    # 简化的使用模式，基于一天中的小时（0-23）
    # 值越小表示自行车空闲率越高，越适合检修
    bike_usage_pattern = {
        0: 0.1, 1: 0.05, 2: 0.02, 3: 0.01, 4: 0.01, 5: 0.05,
        6: 0.2, 7: 0.5, 8: 0.8, 9: 0.7, 10: 0.6, 11: 0.7,
        12: 0.9, 13: 0.6, 14: 0.7, 15: 0.7, 16: 0.6, 17: 0.8,
        18: 0.9, 19: 0.7, 20: 0.5, 21: 0.4, 22: 0.3, 23: 0.2
    }

    best_start, best_end = find_best_inspection_time(schedule, bike_usage_pattern)

    print(f"\n最佳巡检时间建议: 从 {best_start}:00 开始，到 {best_end}:00 结束")

    # 7. 可视化检修路径
    visualize_repair_route(schedule, locations)


if __name__ == "__main__":
    main()
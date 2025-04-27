import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
from scipy.optimize import minimize

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # To display Chinese characters
mpl.rcParams['axes.unicode_minus'] = False  # To prevent negative signs from showing as squares

# 1. Data Preparation
def create_sample_data():
    times = ['7:00', '9:00', '12:00', '14:00', '18:00', '21:00', '23:00']
    locations = ['东门', '南门', '北门', '一食堂', '二食堂', '三食堂', 
                 '梅苑1栋', '菊苑1栋', '教学2楼', '教学4楼', '计算机学院', 
                 '工程中心', '网球场', '体育馆', '校医院']

    data = {
        '东门': [31, 73, 56, 84, 36, 94, 14],
        '南门': [46, 72, 45, 29, 108, 0, 46],
        '北门': [18, 55, 78, 63, 90, 33, 0],
        '一食堂': [0, 21, 45, 8, 0, 36, 82],
        '二食堂': [105, 21, 92, 10, 68, 62, 114],
        '三食堂': [0, 27, 31, 12, 58, 52, 123],
        '梅苑1栋': [97, 34, 46, 22, 46, 36, 128],
        '菊苑1栋': [102, 93, 72, 67, 0, 92, 126],
        '教学2楼': [19, 144, 62, 82, 25, 90, 30],
        '教学4楼': [30, 103, 120, 124, 6, 56, 20],
        '计算机学院': [2, 28, 18, 61, 36, 27, 17],
        '工程中心': [0, 53, 21, 69, 29, 63, 50],
        '网球场': [8, 25, 11, 19, 42, 2, 16],
        '体育馆': [0, 13, 6, 15, 42, 6, 0],
        '校医院': [11, 16, 14, 18, 4, 3, 11]
    }

    df = pd.DataFrame(data, index=times)
    return df

# 2. Model for Evaluating Bike Operation Efficiency
class BikeOperationModel:
    def __init__(self, distribution_data):
        self.distribution_data = distribution_data
        self.times = distribution_data.index.tolist()
        self.locations = distribution_data.columns.tolist()

        # Calculate distance matrix (mock implementation for simplicity)
        self.distance_matrix = self._calculate_distance_matrix()

        # Calculate demand matrix
        self.demand_matrix = self._calculate_demand_matrix()
        
        # 使用需求分析来指导优化
        self.location_demand_analysis = self._analyze_location_demand()

    def _calculate_distance_matrix(self):
        """A mock implementation to simulate distance calculation."""
        n = len(self.locations)
        
        # 创建一个更真实的距离矩阵
        # 假设位置是有一定模式的：食堂周围，宿舍区域，教学区域等
        np.random.seed(42)  # 为了结果可重现
        distance_matrix = np.zeros((n, n))
        
        # 定义几个位置组
        gates = [0, 1, 2]  # 东南北门
        canteens = [3, 4, 5]  # 食堂
        dorms = [6, 7]  # 宿舍
        teaching = [8, 9, 10, 11]  # 教学楼
        sports = [12, 13]  # 体育设施
        others = [14]  # 其他
        
        groups = [gates, canteens, dorms, teaching, sports, others]
        
        # 组内距离较近，组间距离较远
        for i in range(n):
            for j in range(i+1, n):
                # 找出i和j分别属于哪个组
                group_i = None
                group_j = None
                for idx, group in enumerate(groups):
                    if i in group:
                        group_i = idx
                    if j in group:
                        group_j = idx
                
                # 同组内距离较小，不同组距离较大
                if group_i == group_j:
                    distance = np.random.uniform(1, 3)
                else:
                    # 组间距离基于组的远近
                    base_distance = abs(group_i - group_j) * 2
                    distance = np.random.uniform(base_distance + 2, base_distance + 5)
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix

    def _calculate_demand_matrix(self):
        """A mock implementation to simulate demand matrix calculation."""
        demand_matrix = np.zeros((len(self.times) - 1, len(self.locations)))

        for t in range(len(self.times) - 1):
            for i, loc in enumerate(self.locations):
                change = self.distribution_data.iloc[t + 1][loc] - self.distribution_data.iloc[t][loc]
                if change < 0:
                    demand_matrix[t, i] = abs(change)

        return demand_matrix
    
    def _analyze_location_demand(self):
        """分析各位置的需求模式"""
        demand_patterns = {}
        
        for i, loc in enumerate(self.locations):
            # 计算平均需求
            total_demand = sum(self.demand_matrix[:, i])
            avg_demand = total_demand / len(self.times) if len(self.times) > 0 else 0
            
            # 计算需求高峰时段
            time_demand = []
            for t in range(len(self.times) - 1):
                time_demand.append((t, self.demand_matrix[t, i]))
            
            # 按需求量排序
            time_demand.sort(key=lambda x: x[1], reverse=True)
            peak_times = [self.times[td[0]] for td in time_demand[:2] if td[1] > 0]
            
            # 存储结果
            demand_patterns[loc] = {
                'avg_demand': avg_demand,
                'total_demand': total_demand,
                'peak_times': peak_times
            }
        
        return demand_patterns

    def evaluate_efficiency(self):
        """Evaluate bike operation efficiency"""
        avg_availability = self.distribution_data.mean().mean() / self.distribution_data.sum().sum() * len(self.locations)
        dispatch_cost = self._calculate_dispatch_cost()
        demand_satisfaction = self._calculate_demand_satisfaction()
        distribution_balance = self._calculate_distribution_balance()
        utilization_rate = self._calculate_utilization_rate()

        weights = {
            'availability': 0.2,
            'dispatch_cost': 0.3,
            'demand_satisfaction': 0.25,
            'distribution_balance': 0.15,
            'utilization_rate': 0.1
        }

        normalized_dispatch_cost = 1 - (dispatch_cost / 1000)
        score = (weights['availability'] * avg_availability +
                 weights['dispatch_cost'] * normalized_dispatch_cost +
                 weights['demand_satisfaction'] * demand_satisfaction +
                 weights['distribution_balance'] * distribution_balance +
                 weights['utilization_rate'] * utilization_rate)

        return {
            'score': score,
            'availability': avg_availability,
            'dispatch_cost': dispatch_cost,
            'demand_satisfaction': demand_satisfaction,
            'distribution_balance': distribution_balance,
            'utilization_rate': utilization_rate
        }

    def _calculate_dispatch_cost(self):
        total_cost = 0
        for t in range(len(self.times) - 1):
            current_bikes = self.distribution_data.iloc[t].values
            next_bikes = self.distribution_data.iloc[t + 1].values

            surplus = []
            shortage = []

            for i, (curr, next_val) in enumerate(zip(current_bikes, next_bikes)):
                diff = curr - next_val - self.demand_matrix[t, i]
                if diff > 0:
                    surplus.append((i, diff))
                elif diff < 0:
                    shortage.append((i, -diff))

            surplus.sort(key=lambda x: x[1], reverse=True)
            shortage.sort(key=lambda x: x[1], reverse=True)

            for i, s_amount in surplus:
                remaining = s_amount
                for j, d_amount in shortage:
                    if d_amount > 0 and remaining > 0:
                        dispatch_amount = min(remaining, d_amount)
                        remaining -= dispatch_amount
                        shortage[shortage.index((j, d_amount))] = (j, d_amount - dispatch_amount)
                        total_cost += dispatch_amount * self.distance_matrix[i, j]
                        if remaining == 0:
                            break

        return total_cost

    def _calculate_demand_satisfaction(self):
        total_demand = np.sum(self.demand_matrix)
        if total_demand == 0:
            return 1.0
        satisfied_demand = 0
        for t in range(len(self.times) - 1):
            current_bikes = self.distribution_data.iloc[t].values
            for i, loc in enumerate(self.locations):
                demand = self.demand_matrix[t, i]
                if demand <= current_bikes[i]:
                    satisfied_demand += demand
                else:
                    satisfied_demand += current_bikes[i]
        return satisfied_demand / total_demand

    def _calculate_distribution_balance(self):
        mean_distribution = np.mean([np.mean(self.distribution_data[loc]) for loc in self.locations])
        std_distribution = np.std([np.mean(self.distribution_data[loc]) for loc in self.locations])

        if mean_distribution == 0:
            return 0

        cv = std_distribution / mean_distribution
        balance = max(0, 1 - cv)
        return balance

    def _calculate_utilization_rate(self):
        utilization = []
        for t in range(len(self.times) - 1):
            total_bikes = self.distribution_data.iloc[t].sum()
            total_demand = np.sum(self.demand_matrix[t])

            if total_bikes > 0:
                utilization.append(total_demand / total_bikes)
            else:
                utilization.append(0)
        return np.mean(utilization)
        
    # 新增: 优化布局
    def optimize_layout(self):
        """优化单车的初始布局以提高整体效率"""
        print("正在优化单车布局...")
        
        # 获取总单车数，保持不变
        total_bikes = self.distribution_data.iloc[0].sum()
        
        # 获取地点数
        n_locations = len(self.locations)
        
        # 分析原始需求模式
        location_demands = {}
        peak_time_demands = {}
        
        # 计算每个地点的平均需求和需求波动
        for i, loc in enumerate(self.locations):
            # 计算从此位置流出的单车需求
            outflow = [max(0, self.distribution_data.iloc[t][loc] - self.distribution_data.iloc[t+1][loc]) 
                      for t in range(len(self.times)-1)]
            
            # 计算从此位置流出的单车需求总数和平均数
            total_outflow = sum(outflow)
            avg_outflow = total_outflow / len(outflow) if outflow else 0
            
            # 计算需要重新分配的单车数量
            location_demands[loc] = total_outflow
            
            # 统计各时段需求
            for t in range(len(self.times)-1):
                time_key = f"{self.times[t]}-{self.times[t+1]}"
                if time_key not in peak_time_demands:
                    peak_time_demands[time_key] = {}
                peak_time_demands[time_key][loc] = outflow[t]
        
        # 基于需求模式优化初始分布
        # 更加智能的优化策略：根据每个点位的单车使用情况调整分配
        optimized_distribution = [0] * n_locations
        
        # 计算每个位置的优先级得分（需求量 / 输入量）
        priority_scores = []
        for i, loc in enumerate(self.locations):
            # 计算需求
            demand = location_demands[loc]
            
            # 计算单车的最小输入量
            min_inflow = min([self.distribution_data.iloc[t+1][loc] for t in range(len(self.times)-1)])
            
            # 计算优先级得分（需求/输入的比率）
            # 如果输入为0，则设置一个较高优先级
            if min_inflow > 0:
                priority = demand / min_inflow
            else:
                priority = demand * 2 if demand > 0 else 0
                
            priority_scores.append((i, priority))
            
        # 按优先级排序（需求/输入比率越高，优先级越高）
        priority_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 先为每个地点分配最低需求量的单车
        remaining_bikes = total_bikes
        for i, loc in enumerate(self.locations):
            # 找出该地点在各时段的最大需求
            max_demand = max([self.demand_matrix[t, i] for t in range(len(self.times)-1)], default=0)
            
            # 分配单车，但确保不超过总数的一定比例
            base_allocation = min(max_demand * 1.5, remaining_bikes * 0.2)
            optimized_distribution[i] = int(base_allocation)
            remaining_bikes -= optimized_distribution[i]
        
        # 根据优先级分配剩余单车
        for i, priority in priority_scores:
            if remaining_bikes <= 0:
                break
                
            # 基于优先级计算应分配的单车数
            if priority > 0:
                # 计算该位置的分配比例
                allocation_proportion = priority / sum([p for _, p in priority_scores])
                extra_bikes = int(allocation_proportion * remaining_bikes * 0.8)  # 只分配80%的剩余单车
                
                # 确保分配的单车数不超过剩余单车数
                extra_bikes = min(extra_bikes, remaining_bikes)
                
                # 更新分配和剩余单车数
                optimized_distribution[i] += extra_bikes
                remaining_bikes -= extra_bikes
        
        # 如果还有剩余，将它们分配到需求最高的地点
        if remaining_bikes > 0:
            for i, _ in sorted(enumerate(location_demands.values()), key=lambda x: x[1], reverse=True):
                if remaining_bikes <= 0:
                    break
                optimized_distribution[i] += 1
                remaining_bikes -= 1
        
        # 创建新的优化布局数据
        optimized_layout = self.distribution_data.copy()
        optimized_layout.iloc[0] = optimized_distribution
        
        return optimized_layout

# 3. Visualize the Data: Generate the heatmap
def visualize_data(distribution_data, title="共享单车停车点布局 - 时间分布热力图"):
    plt.figure(figsize=(14, 8))
    sns.heatmap(distribution_data.T, cmap="YlOrRd", annot=True, fmt="d", linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 新增：创建气泡图
def create_bubble_chart(distribution_data, title="共享单车分布气泡图"):
    # 准备数据
    locations = distribution_data.columns
    times = distribution_data.index
    
    # 创建坐标网格
    time_indices = list(range(len(times)))
    location_indices = list(range(len(locations)))
    
    x = []
    y = []
    sizes = []
    labels = []
    
    # 填充数据
    for i, time in enumerate(times):
        for j, location in enumerate(locations):
            bike_count = distribution_data.loc[time, location]
            if bike_count > 0:  # 只显示有单车的地点
                x.append(i)
                y.append(j)
                sizes.append(bike_count * 10)  # 放大尺寸使气泡更明显
                labels.append(f"{location}:{bike_count}")
    
    # 创建气泡图
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(x, y, s=sizes, alpha=0.5, c=sizes, cmap='viridis', edgecolors='grey')
    
    # 添加标签和标题
    plt.title(title, fontsize=16)
    plt.xlabel("时间", fontsize=14)
    plt.ylabel("地点", fontsize=14)
    
    # 设置刻度标签
    plt.xticks(time_indices, times, rotation=45)
    plt.yticks(location_indices, locations)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('单车数量', fontsize=12)
    
    # 为部分重要气泡添加标签
    threshold = np.percentile(sizes, 75)  # 仅标注较大的气泡
    for i, txt in enumerate(labels):
        if sizes[i] > threshold:
            plt.annotate(txt.split(':')[1], (x[i], y[i]), 
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8)
    
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# 新增：对比展示优化前后的布局
def compare_layouts(original_data, optimized_data):
    """对比展示优化前后的初始布局"""
    # 创建数据框
    comparison = pd.DataFrame({
        '原始布局': original_data.iloc[0],
        '优化布局': optimized_data.iloc[0],
        '变化量': optimized_data.iloc[0] - original_data.iloc[0]
    })
    
    # 计算变化百分比
    comparison['变化百分比'] = comparison['变化量'] / comparison['原始布局'] * 100
    comparison['变化百分比'] = comparison['变化百分比'].fillna(0)
    
    # 对比可视化 - 柱状图
    plt.figure(figsize=(16, 8))
    comparison[['原始布局', '优化布局']].plot(kind='bar', figsize=(16, 8))
    plt.title('优化前后停车点布局对比', fontsize=16)
    plt.xlabel('停车点', fontsize=14)
    plt.ylabel('单车数量', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # 对比可视化 - 饼图显示单车分布变化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 原始分布
    ax1.pie(original_data.iloc[0], labels=original_data.columns, autopct='%1.1f%%', startangle=90, 
            textprops={'fontsize': 8})
    ax1.set_title('原始单车分布', fontsize=14)
    
    # 优化后分布
    ax2.pie(optimized_data.iloc[0], labels=optimized_data.columns, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 8})
    ax2.set_title('优化后单车分布', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    # 返回对比数据
    return comparison

# 新增：显示优化前后效率差异的雷达图
def plot_efficiency_radar(original_efficiency, optimized_efficiency):
    # 准备数据
    metrics = ['可用率', '调度成本', '需求满足率', '点位分布均衡性', '使用率']
    
    # 提取值并标准化（对调度成本进行反向处理，因为越低越好）
    original_values = [
        original_efficiency['availability'],
        1 - (original_efficiency['dispatch_cost'] / 1000),  # 标准化
        original_efficiency['demand_satisfaction'],
        original_efficiency['distribution_balance'],
        original_efficiency['utilization_rate']
    ]
    
    optimized_values = [
        optimized_efficiency['availability'],
        1 - (optimized_efficiency['dispatch_cost'] / 1000),  # 标准化
        optimized_efficiency['demand_satisfaction'],
        optimized_efficiency['distribution_balance'],
        optimized_efficiency['utilization_rate']
    ]
    
    # 计算角度
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 添加闭合的值
    original_values += original_values[:1]
    optimized_values += optimized_values[:1]
    metrics += metrics[:1]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 绘制原始效率
    ax.plot(angles, original_values, 'o-', linewidth=2, label='原始布局')
    ax.fill(angles, original_values, alpha=0.25)
    
    # 绘制优化后效率
    ax.plot(angles, optimized_values, 'o-', linewidth=2, label='优化布局')
    ax.fill(angles, optimized_values, alpha=0.25)
    
    # 设置标签
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics[:-1])
    
    # 添加图例和标题
    ax.legend(loc='upper right')
    ax.set_title('优化前后效率指标对比', fontsize=15)
    
    # 调整网格线
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# 4. Main function to evaluate and visualize efficiency
def solve_problem3():
    print("问题3：建立共享单车运营效率的评价模型")

    # Prepare data
    distribution_data = create_sample_data()

    # Create the model
    bike_model = BikeOperationModel(distribution_data)

    # Evaluate efficiency
    current_efficiency = bike_model.evaluate_efficiency()
    print("\n当前共享单车运营效率评估结果：")
    print(f"综合评分: {current_efficiency['score']:.4f}")
    print(f"可用率: {current_efficiency['availability']:.4f}")
    print(f"调度成本: {current_efficiency['dispatch_cost']:.2f}")
    print(f"需求满足率: {current_efficiency['demand_satisfaction']:.4f}")
    print(f"点位分布均衡性: {current_efficiency['distribution_balance']:.4f}")
    print(f"使用率: {current_efficiency['utilization_rate']:.4f}")

    # Visualize current layout
    visualize_data(distribution_data, "原始共享单车停车点布局 - 时间分布热力图")
    
    # 使用气泡图可视化数据
    create_bubble_chart(distribution_data, "原始共享单车分布气泡图")
    
    # 优化布局
    optimized_layout = bike_model.optimize_layout()
    
    # 评估优化后的效率
    optimized_model = BikeOperationModel(optimized_layout)
    optimized_efficiency = optimized_model.evaluate_efficiency()
    
    print("\n优化后共享单车运营效率评估结果：")
    print(f"综合评分: {optimized_efficiency['score']:.4f}")
    print(f"可用率: {optimized_efficiency['availability']:.4f}")
    print(f"调度成本: {optimized_efficiency['dispatch_cost']:.2f}")
    print(f"需求满足率: {optimized_efficiency['demand_satisfaction']:.4f}")
    print(f"点位分布均衡性: {optimized_efficiency['distribution_balance']:.4f}")
    print(f"使用率: {optimized_efficiency['utilization_rate']:.4f}")
    
    # 优化效果对比
    print("\n优化效果：")
    improvement = (optimized_efficiency['score'] - current_efficiency['score']) / current_efficiency['score'] * 100
    print(f"综合评分提升：{improvement:.2f}%")
    
    # 可视化优化后的布局
    visualize_data(optimized_layout, "优化后共享单车停车点布局 - 时间分布热力图")
    create_bubble_chart(optimized_layout, "优化后共享单车分布气泡图")
    
    # 对比优化前后的布局
    comparison = compare_layouts(distribution_data, optimized_layout)
    print("\n优化前后布局对比：")
    print(comparison[comparison['变化量'] != 0].sort_values('变化量', ascending=False))
    
    # 显示雷达图比较优化前后的效率
    plot_efficiency_radar(current_efficiency, optimized_efficiency)

if __name__ == "__main__":
    solve_problem3()

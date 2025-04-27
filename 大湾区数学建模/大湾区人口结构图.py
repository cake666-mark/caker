import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 检查字体路径是否正确
font_prop = font_manager.FontProperties(fname=font_path)

# 加载Excel文件
file_path = r"粤港澳湾区人口.xlsx"
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 打印数据前几行和列名，确保数据加载正确
print(data.columns)  # 检查列名
print(data.head())    # 查看数据前几行

# 确保年份列的数据类型为整数或浮动
years = data['年份/人口(万人)'].astype(int)  # 强制转换为整数类型，避免数据类型问题

# 检查年份数据是否正确
print(years)

# 获取城市名，排除第一列
cities = data.columns[1:]

# 创建图形
plt.figure(figsize=(12, 8))

# 循环绘制每个城市的人口变化曲线
for city in cities:
    plt.plot(years, data[city], label=city)  # 绘制曲线
    plt.text(years.iloc[-1], data[city].iloc[-1], city, fontsize=9, va='center', ha='left', fontproperties=font_prop)

# 设置 x 轴和 y 轴的标签和图表标题
plt.xlabel('年份', fontsize=12, fontproperties=font_prop)
plt.ylabel('人口 (万人)', fontsize=12, fontproperties=font_prop)
plt.title('粤港澳大湾区各城市人口变化', fontsize=14, fontproperties=font_prop)
plt.grid(True)

# 确保 x 轴和 y 轴可见
plt.xticks(fontsize=10, fontproperties=font_prop)
plt.yticks(fontsize=10, fontproperties=font_prop)

# 调整布局，避免重叠
plt.tight_layout()

# 显示图表
plt.show()

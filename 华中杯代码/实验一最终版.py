# -*- coding: utf‑8 -*-
"""
根据《附件1‑共享单车分布统计表.xlsx》生成“表1：各停车点在 7:00‑23:00
七个整点的平均车辆数”并导出为 Excel。
"""

import pandas as pd
import datetime as dt
from pathlib import Path

# === 0  路径设置 =============================================================
INPUT_FILE = Path(r'C:\Users\KING\Desktop\附件1-共享单车分布统计表.xlsx')  # 原始数据
OUTPUT_FILE = Path(r'C:\Users\KING\Desktop\共享单车分布统计表_表1.xlsx')  # 结果表 1

# === 1  读取原始数据 ==========================================================
df = pd.read_excel(INPUT_FILE, engine='openpyxl')

# 把无标题列重命名（原文件第 1、2 列分别是星期与时间）
df = df.rename(columns={'Unnamed: 0': 'weekday',
                        'Unnamed: 1': 'time'})

# 填充缺失星期（向下复制上一行的星期）
df['weekday'] = df['weekday'].ffill()

# === 2  预处理数值 & 时间映射 ===============================================
# 把类似 “200+” 的字符串统一改为 200；空白统一按 NaN 处理
def convert_val(x):
    if isinstance(x, str) and '+' in x:
        return float(x.replace('+', ''))
    return x
df = df.applymap(convert_val)

# 定义目标整点
TARGET_TIMES = [dt.time(7, 0), dt.time(9, 0), dt.time(12, 0),
                dt.time(14, 0), dt.time(18, 0), dt.time(21, 0),
                dt.time(23, 0)]

def map_to_nearest(t):
    """把原始时间映射到最近的目标整点"""
    return min(TARGET_TIMES,
               key=lambda x: abs(
                   dt.datetime.combine(dt.date(1, 1, 1), x)
                   - dt.datetime.combine(dt.date(1, 1, 1), t))
              )

df['mapped_time'] = df['time'].apply(map_to_nearest)

# === 3  生成表 1（平均车辆数矩阵） ==========================================
SPOTS = ['东门', '南门', '北门', '一食堂', '二食堂', '三食堂',
         '梅苑1栋', '菊苑1栋', '教学2楼', '教学4楼',
         '计算机学院', '工程中心', '网球场', '体育馆', '校医院']

# 按 mapped_time 分组，对同一停车点取各日平均
table1 = (df
          .groupby('mapped_time')[SPOTS]
          .mean()
          .round(1)        # 保留 1 位小数
          .fillna(0)       # 缺失记 0
          .round(0)        # 四舍五入取整
          .astype(int)     # 转为 int 显示
         )

# （可选）加一列“total”快速查看全校车量
table1['total'] = table1.sum(axis=1)

# === 4  导出 Excel ==========================================================
table1.to_excel(OUTPUT_FILE, engine='openpyxl')
print(f"结果已导出到 {OUTPUT_FILE}")
# === 5  绘制折线图 ==========================================================
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为 SimHei 以支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 将表 1 转换为适合绘图的格式
table1.reset_index(inplace=True)  # 将索引重置为普通列，方便绘图

plt.figure(figsize=(12, 8))

# 将 mapped_time 转换为字符串格式，便于 Matplotlib 处理
table1['mapped_time'] = table1['mapped_time'].apply(lambda x: x.strftime('%H:%M'))

plt.figure(figsize=(12, 8))

# 为每个停车点绘制折线
for spot in SPOTS:
    plt.plot(table1['mapped_time'], table1[spot], marker='o', label=spot)

# 添加图例、标题和坐标轴标签
plt.title("各停车点在不同时间的平均车辆数分布", fontsize=16)
plt.xlabel("时间", fontsize=12)
plt.ylabel("平均车辆数", fontsize=12)
plt.xticks(rotation=45)  # 旋转 x 轴刻度
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 图例放在图外
plt.grid(True)

# 显示图表
plt.tight_layout()  # 自动调整布局以防止重叠
plt.show()
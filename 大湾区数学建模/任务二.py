import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 设置支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 读取数据
gdp_data = pd.read_excel('GDP总值（单位亿元）.xlsx')
industry_data = pd.read_excel('产业（工厂总收入）.xlsx')
unemployment_data = pd.read_excel('失业率.xlsx')
employment_data = pd.read_excel('就业率.xlsx')
tech_investment_data = pd.read_excel('科技总投入.xlsx')
population_data = pd.read_excel('粤港澳湾区人口.xlsx')

# 假设数据按年份排列，提取需要的列
data = pd.DataFrame({
    'Year': gdp_data['year'],  # GDP数据的年份列
    'GDP': gdp_data[['广州', '深圳', '珠海', '佛山', '惠州', '东莞', '中山', '江门', '肇庆', '香港', '澳门']].sum(axis=1),  # 所有城市 GDP 的总和
    'Industry': industry_data[['广州', '深圳', '珠海', '佛山', '惠州', '东莞', '中山', '江门', '肇庆', '香港', '澳门']].sum(axis=1),  # 所有城市产业收入的总和
    'Tech_Investment': tech_investment_data[['广州(亿元)', '深圳(亿元)', '珠海(亿元)', '佛山(亿元)', '惠州(亿元)', '东莞(亿元)', '中山(亿元)', '江门(亿元)', '肇庆(亿元)', '香港(亿元)', '澳门(亿元)']].sum(axis=1),  # 所有城市科技投入的总和
    'Population': population_data[['广州', '深圳', '珠海', '佛山', '惠州', '东莞', '中山', '江门', '肇庆', '香港', '澳门']].sum(axis=1)  # 所有城市人口的总和
})

# 自变量和因变量
X = data[['Industry', 'Tech_Investment', 'Population']]
y = data['GDP']

# 训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 打印模型的系数
print("线性回归模型的系数：")
print(f"常数项 (β0): {model.intercept_}")
print(f"产业收入系数 (β1): {model.coef_[0]}")
print(f"科技投入系数 (β2): {model.coef_[1]}")
print(f"人口系数 (β3): {model.coef_[2]}")

# 未来5-10年的预测（2023-2030年）
future_years = np.arange(2023, 2031)
predicted_industry = [data['Industry'].iloc[-1] * (1 + 0.01)**i for i in range(8)]  # 假设未来产业年增长1%
predicted_tech_investment = [data['Tech_Investment'].iloc[-1] * (1 + 0.02)**i for i in range(8)]  # 假设科技投入年增长1%
predicted_population = [data['Population'].iloc[-1] * (1 + 0.01)**i for i in range(8)]  # 假设人口年增长1%

# 创建未来预测DataFrame
future_data = pd.DataFrame({
    'Year': future_years,
    'Industry': predicted_industry,
    'Tech_Investment': predicted_tech_investment,
    'Population': predicted_population
})

# 进行GDP预测
future_X = future_data[['Industry', 'Tech_Investment', 'Population']]
future_gdp_predictions = model.predict(future_X)

# 打印预测的GDP值
print("\n未来几年（2023-2030年）预测的GDP值：")
for year, gdp in zip(future_years, future_gdp_predictions):
    print(f"{year}年: {gdp:.2f} 亿元")

# 整合历史与预测数据
all_years = np.concatenate((data['Year'].values, future_years))
all_gdp = np.concatenate((data['GDP'].values, future_gdp_predictions))

# 可视化
plt.figure(figsize=(12, 6))
sns.lineplot(x=all_years, y=all_gdp, marker='o', label='GDP 预测')
plt.axvline(x=2022, color='red', linestyle='--', label='预测开始')  # 预测开始的虚线
plt.title('粤港澳大湾区 GDP 预测')
plt.xlabel('年份')
plt.ylabel('GDP（亿元）')
plt.legend(title='图例')
plt.grid()

# 显示图表
plt.show()

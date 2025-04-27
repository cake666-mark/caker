import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 设置支持中文的字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 读取数据
gdp_data = pd.read_excel('GDP总值（单位亿元）.xlsx')
industry_data = pd.read_excel('产业（工厂总收入）.xlsx')
unemployment_data = pd.read_excel('失业率.xlsx')
employment_data = pd.read_excel('就业率.xlsx')
tech_investment_data = pd.read_excel('科技总投入.xlsx')
population_data = pd.read_excel('粤港澳湾区人口.xlsx')

tokyo_data = pd.read_excel('Tokyo.xlsx')
nyc_data = pd.read_excel('NYC.xlsx')

# 去除列名中的多余空格
tokyo_data.columns = tokyo_data.columns.str.strip()
nyc_data.columns = nyc_data.columns.str.strip()

# 提取相关列
# Tokyo
tokyo_years = tokyo_data['年份']
tokyo_gdp = tokyo_data['GDP总值（估算，亿元）']
tokyo_population = tokyo_data['人口（百万）']
tokyo_tech_investment = tokyo_data['科技总投入（估算，亿元）']
tokyo_industry_income = tokyo_data['工厂总收入（估算，亿元）']

# NYC
nyc_years = nyc_data['年份']
nyc_gdp = nyc_data['GDP总值（估算，亿元）']
nyc_population = nyc_data['人口（百万）']
nyc_tech_investment = nyc_data['科技总投入（估算，亿元）']
nyc_industry_income = nyc_data['工厂总收入（估算，亿元）']

# 特征和标签
tokyo_features = pd.DataFrame({
    'Population': tokyo_population,
    'TechInvestment': tokyo_tech_investment,
    'IndustryIncome': tokyo_industry_income
})
nyc_features = pd.DataFrame({
    'Population': nyc_population,
    'TechInvestment': nyc_tech_investment,
    'IndustryIncome': nyc_industry_income
})

# 构建和训练模型
tokyo_model = LinearRegression().fit(tokyo_features, tokyo_gdp)
nyc_model = LinearRegression().fit(nyc_features, nyc_gdp)

# 预测未来5-10年（2024-2033）
future_years = np.arange(2024, 2034)
future_population_tokyo = np.linspace(tokyo_population.iloc[-1], tokyo_population.iloc[-1] * 1.02, len(future_years))
future_tech_investment_tokyo = np.linspace(tokyo_tech_investment.iloc[-1], tokyo_tech_investment.iloc[-1] * 1.05, len(future_years))
future_industry_income_tokyo = np.linspace(tokyo_industry_income.iloc[-1], tokyo_industry_income.iloc[-1] * 1.03, len(future_years))

future_population_nyc = np.linspace(nyc_population.iloc[-1], nyc_population.iloc[-1] * 1.02, len(future_years))
future_tech_investment_nyc = np.linspace(nyc_tech_investment.iloc[-1], nyc_tech_investment.iloc[-1] * 1.05, len(future_years))
future_industry_income_nyc = np.linspace(nyc_industry_income.iloc[-1], nyc_industry_income.iloc[-1] * 1.03, len(future_years))

# 构建未来的特征数据
future_features_tokyo = pd.DataFrame({
    'Population': future_population_tokyo,
    'TechInvestment': future_tech_investment_tokyo,
    'IndustryIncome': future_industry_income_tokyo
})
future_features_nyc = pd.DataFrame({
    'Population': future_population_nyc,
    'TechInvestment': future_tech_investment_nyc,
    'IndustryIncome': future_industry_income_nyc
})

# 进行预测
future_gdp_tokyo = tokyo_model.predict(future_features_tokyo)
future_gdp_nyc = nyc_model.predict(future_features_nyc)

# 读取粤港澳数据
data = pd.DataFrame({
    'Year': gdp_data['year'],
    'GDP': gdp_data[['广州', '深圳', '珠海', '佛山', '惠州', '东莞', '中山', '江门', '肇庆', '香港', '澳门']].sum(axis=1),
    'Industry': industry_data[['广州', '深圳', '珠海', '佛山', '惠州', '东莞', '中山', '江门', '肇庆', '香港', '澳门']].sum(axis=1),
    'Tech_Investment': tech_investment_data[['广州(亿元)', '深圳(亿元)', '珠海(亿元)', '佛山(亿元)', '惠州(亿元)', '东莞(亿元)', '中山(亿元)', '江门(亿元)', '肇庆(亿元)', '香港(亿元)', '澳门(亿元)']].sum(axis=1),
    'Population': population_data[['广州', '深圳', '珠海', '佛山', '惠州', '东莞', '中山', '江门', '肇庆', '香港', '澳门']].sum(axis=1)
})

# 自变量和因变量
X = data[['Industry', 'Tech_Investment', 'Population']]
y = data['GDP']

# 训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 未来5-10年的预测（2024-2033年）
future_population = np.linspace(data['Population'].iloc[-1], data['Population'].iloc[-1] * 1.02, len(future_years))
future_tech_investment = np.linspace(data['Tech_Investment'].iloc[-1], data['Tech_Investment'].iloc[-1] * 1.05, len(future_years))
future_industry = np.linspace(data['Industry'].iloc[-1], data['Industry'].iloc[-1] * 1.03, len(future_years))

# 创建未来预测DataFrame
future_data = pd.DataFrame({
    'Year': future_years,
    'Industry': future_industry,
    'Tech_Investment': future_tech_investment,
    'Population': future_population
})

# 进行GDP预测
future_X = future_data[['Industry', 'Tech_Investment', 'Population']]
future_gdp_predictions = model.predict(future_X)

# 整合历史与预测数据
all_years = np.concatenate((data['Year'].values, future_years))
all_gdp = np.concatenate((data['GDP'].values, future_gdp_predictions))

# 可视化
plt.figure(figsize=(12, 8))

# 绘制粤港澳大湾区历史数据和预测数据
plt.plot(data['Year'], data['GDP'], label="粤港澳大湾区 历史 GDP", color='orange', marker='o')  # 历史数据
plt.plot(all_years, all_gdp, label="粤港澳大湾区 GDP 预测", color='orange', linestyle='--', marker='x')  # 预测数据

# 绘制东京 GDP 历史数据和预测数据
plt.plot(tokyo_years, tokyo_gdp, label="东京历史 GDP", color='blue', linestyle='--', marker='o')
plt.plot(future_years, future_gdp_tokyo, label="东京预测 GDP", color='blue', marker='x')

# 绘制纽约 GDP 历史数据和预测数据
plt.plot(nyc_years, nyc_gdp, label="纽约历史 GDP", color='green', linestyle='--', marker='o')
plt.plot(future_years, future_gdp_nyc, label="纽约预测 GDP", color='green', marker='x')

plt.xlabel("年份")
plt.ylabel("GDP（亿元）")
plt.title("粤港澳大湾区、东京与纽约 GDP 预测（2024-2033年）")
plt.legend()
plt.grid(True)
plt.show()

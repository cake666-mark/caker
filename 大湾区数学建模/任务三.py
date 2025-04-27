import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 读取数据
tokyo_data = pd.read_excel('Tokyo.xlsx')
nyc_data = pd.read_excel('NYC.xlsx')

# 打印列名以检查是否有 '年份' 列
print(tokyo_data.columns)
print(nyc_data.columns)

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

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(tokyo_years, tokyo_gdp, label="Tokyo Historical GDP", color='blue', linestyle='--', marker='o')
plt.plot(nyc_years, nyc_gdp, label="NYC Historical GDP", color='green', linestyle='--', marker='o')
plt.plot(future_years, future_gdp_tokyo, label="Tokyo Predicted GDP", color='blue', marker='x')
plt.plot(future_years, future_gdp_nyc, label="NYC Predicted GDP", color='green', marker='x')

plt.xlabel("Year")
plt.ylabel("GDP (in billion USD)")
plt.title("Tokyo and NYC GDP Prediction (2024-2033)")
plt.legend()
plt.grid(True)
plt.show()

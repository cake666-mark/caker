import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# 读取数据
gdp_data = pd.read_excel('GDP总值（单位亿元）.xlsx')
industry_data = pd.read_excel('产业（工厂总收入）.xlsx')
unemployment_rate_data = pd.read_excel('失业率.xlsx')
employment_rate_data = pd.read_excel('就业率.xlsx')
tech_investment_data = pd.read_excel('科技总投入.xlsx')
population_data = pd.read_excel('粤港澳湾区人口.xlsx')

# 合并数据
data = pd.DataFrame({
    'GDP': gdp_data.iloc[:, 0],
    '产业': industry_data.iloc[:, 0],
    '失业率': unemployment_rate_data.iloc[:, 0],
    '就业率': employment_rate_data.iloc[:, 0],
    '科技总投入': tech_investment_data.iloc[:, 0],
    '人口': population_data.iloc[:, 0]
})

# 检查缺失值
print("缺失值统计：")
print(data.isnull().sum())

# 删除包含缺失值的行（或选择填充缺失值）
data = data.dropna()
# 或者填充缺失值
# data.fillna(data.mean(), inplace=True)

# 选择自变量和因变量
X = data[['产业', '失业率', '就业率', '科技总投入', '人口']]
y = data['GDP']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测和评估模型
y_pred = model.predict(X_test)
print('Coefficients:', model.coef_)

# 使用statsmodels获取详细结果
X = sm.add_constant(X)  # 添加常数项
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

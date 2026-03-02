import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

df = pd.read_excel('df_n28.xlsx')
print('数据形状:', df.shape)
print('\n前5行数据:'); print(df.head())
print('\n空值统计:')
print(df.isnull().sum())
print('\n基本统计信息:')
print(df.describe())
# 清洗数据：去除D、I、J、L列为空的行
df_clean = df.dropna(subset=['MD03_DIST_LINE_CODE', 'MD03_RETAIL_CUST_LON', 'MD03_RETAIL_CUST_LAT', 'BB_RTL_CUST_ORDER_WEEKDAY'])
# 清洗数据：去除重复行
df_clean = df_clean.drop_duplicates()
print('\n清洗后数据形状:', df_clean.shape)
# 保存清洗后的数据
df_clean.to_excel('df_n28_cleaned.xlsx', index=False)
print('\n清洗后的数据已保存为 df_n28_cleaned.xlsx')

# 读取清洗后的数据
df = pd.read_excel('df_n28_cleaned.xlsx')
# 仅保留经度≤118且纬度≤38.5的点
df = df[(df['MD03_RETAIL_CUST_LON'] <= 118) & (df['MD03_RETAIL_CUST_LAT'] <= 38.5)]

# 设置图形大小
plt.figure(figsize=(14, 12))

# 使用 seaborn 绘制散点图，按 MD03_DIST_LINE_CODE 区分颜色
sns.scatterplot(
    data=df,
    x='MD03_RETAIL_CUST_LON',
    y='MD03_RETAIL_CUST_LAT',
    hue='MD03_DIST_LINE_CODE',
    palette='tab10',
    edgecolor=None,
    s=10,
    alpha=0.5
)

# 添加标题和坐标轴标签
plt.title('零售客户地理分布（按配送车次区分）')
plt.xlabel('经度')
plt.ylabel('纬度')

# 显示图例
plt.legend(title='配送点', bbox_to_anchor=(1.05, 1), loc='upper left')

# 保存并展示图形
plt.tight_layout()
plt.savefig('零售客户地理分布_按配送线路.png', dpi=300)
plt.show()

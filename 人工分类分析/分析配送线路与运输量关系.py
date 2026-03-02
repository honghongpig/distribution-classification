#分了几类，分类相关性，与计算机分类的区别
import seaborn as sns
import pandas as pd
df_clean=pd.read_excel('df_n28_cleaned.xlsx')
# 查看MD03_DIST_LINE_CODE字段的分类情况
print("MD03_DIST_LINE_CODE 分类数量:", df_clean['MD03_DIST_LINE_CODE'].nunique())
print("MD03_DIST_LINE_CODE 分类值:")
print(df_clean['MD03_DIST_LINE_CODE'].value_counts())
# 每一类各有多少条记录
print("\n每一类分了多少个：")
print(df_clean.groupby('MD03_DIST_LINE_CODE').size().sort_values(ascending=False))
# 每一类对应的 MD03_DELIVER_DELIVER_NO 数量总和
print("\n每一类对应的 MD03_DELIVER_DELIVER_NO 数量总和：")
print(df_clean.groupby('MD03_DIST_LINE_CODE')['MD03_DELIVER_DELIVER_NO'].sum().sort_values(ascending=False))
# 每一类对应的 MD03_DELIVER_DELIVER_NO 数量平均值
print("\n每一类对应的 MD03_DELIVER_DELIVER_NO 数量平均值：")
print(df_clean.groupby('MD03_DIST_LINE_CODE')['MD03_DELIVER_DELIVER_NO'].mean().sort_values(ascending=False))



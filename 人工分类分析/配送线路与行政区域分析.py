import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_excel('df_n28_cleaned.xlsx')

print("=== 配送线路代码与行政区域详细分析 ===")

# 1. 每个区县的配送线路分布
print("\n1. 每个区县的主要配送线路:")
district_line_info = {}
for district in df['district'].unique():
    district_data = df[df['district'] == district]
    if len(district_data) > 0:
        line_counts = district_data['MD03_DIST_LINE_CODE'].value_counts()
        top_line = line_counts.idxmax()
        top_count = line_counts.max()
        total_count = len(district_data)
        top_ratio = top_count / total_count
        district_line_info[district] = {
            '主要配送线路': top_line,
            '占比': top_ratio,
            '配送线路数量': len(line_counts),
            '客户数量': total_count
        }
        print(f"区县: {district}, 主要配送线路: {top_line}, 占比: {top_ratio:.2f}, 配送线路数量: {len(line_counts)}, 客户数量: {total_count}")

# 2. 配送线路覆盖的区县数量分布
print("\n2. 配送线路覆盖的区县数量分布:")
line_district_count = df.groupby('MD03_DIST_LINE_CODE')['district'].nunique()
district_coverage_counts = line_district_count.value_counts()
print(district_coverage_counts)


# 3. 每个配送线路的区县覆盖情况
print("\n5. 部分配送线路的区县覆盖情况（前10个配送线路）:")
top_lines = df['MD03_DIST_LINE_CODE'].value_counts().head(10).index
for line in top_lines:
    line_data = df[df['MD03_DIST_LINE_CODE'] == line]
    districts = line_data['district'].unique()
    print(f"配送线路: {line}, 覆盖区县: {len(districts)}个, 区县列表: {', '.join(districts)}")

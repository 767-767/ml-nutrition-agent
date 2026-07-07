"""
数据预处理脚本：USDA SR Legacy 数据集清洗与特征工程
"""

import pandas as pd
import os

DATA_DIR = "./data"
OUTPUT_PATH = os.path.join(DATA_DIR, "cleaned_food_data.csv")

print("=" * 60)
print("开始数据预处理...")
print("=" * 60)

# 1. 读取原始数据
print("正在读取原始数据...")
food_df = pd.read_csv(os.path.join(DATA_DIR, "food.csv"), encoding='utf-8')
nutrient_df = pd.read_csv(os.path.join(DATA_DIR, "nutrient.csv"), encoding='utf-8')
food_nutrient_df = pd.read_csv(os.path.join(DATA_DIR, "food_nutrient.csv"), encoding='utf-8')

print(f"food.csv: {food_df.shape[0]} 条食物记录")
print(f"nutrient.csv: {nutrient_df.shape[0]} 种营养成分")
print(f"food_nutrient.csv: {food_nutrient_df.shape[0]} 条关联记录")

# 2. 筛选我们需要的营养成分（营养素ID）
TARGET_NUTRIENTS = {
    1008: "energy_kcal",
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbs_g",
    1079: "fiber_g"
}
print(f"\n目标营养成分: {list(TARGET_NUTRIENTS.values())}")

# 3. 筛选出目标营养素的关联数据
mask = food_nutrient_df["nutrient_id"].isin(TARGET_NUTRIENTS.keys())
filtered_food_nutrient = food_nutrient_df[mask].copy()
print(f"筛选后关联记录: {filtered_food_nutrient.shape[0]} 条")

# 4. 添加营养名称列（便于理解）
nutrient_name_map = nutrient_df.set_index("id")["name"].to_dict()
filtered_food_nutrient["nutrient_name"] = filtered_food_nutrient["nutrient_id"].map(nutrient_name_map)

# 5. 透视表：将每种营养变成一列（注意：列名是 fdc_id，不是 food_id）
print("\n正在生成透视表（每个食物一行，每种营养一列）...")
pivot_df = filtered_food_nutrient.pivot_table(
    index="fdc_id",          # ✅ 改成 fdc_id
    columns="nutrient_id",
    values="amount"
).reset_index()

# 6. 重命名列
pivot_df.columns = ["fdc_id"] + [TARGET_NUTRIENTS.get(col, col) for col in pivot_df.columns if col != "fdc_id"]
print(f"透视表完成: {pivot_df.shape[0]} 行 × {pivot_df.shape[1]} 列")

# 7. 合并食物名称（注意：food.csv 里食物的ID列也叫 fdc_id）
food_subset = food_df[["fdc_id", "description"]].copy()
food_subset = food_subset.rename(columns={"description": "food_name"})
merged_df = pivot_df.merge(food_subset, on="fdc_id", how="inner")
print(f"合并食物名称后: {merged_df.shape[0]} 条记录")

# 8. 数据清洗
print("\n正在进行数据清洗...")
null_counts = merged_df[TARGET_NUTRIENTS.values()].isnull().sum()
if null_counts.sum() > 0:
    print(f"发现缺失值: {null_counts[null_counts > 0].to_dict()}")
    merged_df = merged_df.dropna(subset=TARGET_NUTRIENTS.values())
    print(f"删除缺失值后: {merged_df.shape[0]} 条记录")
else:
    print("无缺失值")

merged_df = merged_df[merged_df["energy_kcal"] > 0]
print(f"过滤0热量后: {merged_df.shape[0]} 条记录")

# 9. 特征工程
print("\n正在生成派生特征...")
merged_df["dry_mass_g"] = merged_df["protein_g"] + merged_df["fat_g"] + merged_df["carbs_g"] + merged_df["fiber_g"]
merged_df["energy_density"] = merged_df["energy_kcal"] / (merged_df["dry_mass_g"] + 1)
merged_df["protein_ratio"] = merged_df["protein_g"] / (merged_df["dry_mass_g"] + 1)
merged_df["fat_ratio"] = merged_df["fat_g"] / (merged_df["dry_mass_g"] + 1)
merged_df["carbs_ratio"] = merged_df["carbs_g"] / (merged_df["dry_mass_g"] + 1)
print(f"派生特征已添加: energy_density, protein_ratio, fat_ratio, carbs_ratio")

# 10. 分类标签
def protein_label(row):
    if row["protein_ratio"] > 0.25:
        return "High Protein"
    elif row["protein_ratio"] > 0.15:
        return "Medium Protein"
    else:
        return "Low Protein"

merged_df["protein_category"] = merged_df.apply(protein_label, axis=1)
print(f"分类标签已生成: {merged_df['protein_category'].value_counts().to_dict()}")

# 11. 保存
print(f"\n正在保存清洗后的数据到: {OUTPUT_PATH}")
merged_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print(f"数据预处理完成！最终数据形状: {merged_df.shape[0]} 行 × {merged_df.shape[1]} 列")
print("=" * 60)

print("\n数据预览（前5行）:")
print(merged_df.head())
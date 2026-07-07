"""
模型训练脚本：训练两个经典ML模型
1. 随机森林回归器 → 预测热量（展示回归任务）
2. SVM分类器 → 预测蛋白质等级（展示分类任务）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 配置路径
DATA_PATH = "./data/cleaned_food_data.csv"
MODEL_DIR = "./model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("机器学习模型训练开始")
print("=" * 60)

# 1. 读取数据
print("\n读取清洗后的数据...")
df = pd.read_csv(DATA_PATH)
print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")

# 2. 准备特征和目标变量
# 用于回归的特征：蛋白质、脂肪、碳水、纤维、能量密度
feature_cols = ["protein_g", "fat_g", "carbs_g", "fiber_g", "energy_density"]
X = df[feature_cols].copy()

# 目标1：热量（回归任务）
y_reg = df["energy_kcal"].copy()

# 目标2：蛋白质等级（分类任务）
y_clf = df["protein_category"].copy()

print(f"\n特征矩阵: {X.shape[0]} 行 × {X.shape[1]} 列")
print(f"特征: {feature_cols}")
print(f"\n分类目标分布:")
print(y_clf.value_counts())

# 3. 划分训练集和测试集（80% / 20%）
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

print(f"\n训练集: {X_train.shape[0]} 条")
print(f"测试集: {X_test.shape[0]} 条")

# ============================================================
# 模型1：随机森林回归器（预测热量）
# ============================================================
print("\n" + "=" * 60)
print("模型1：随机森林回归器（预测热量）")
print("=" * 60)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# 训练
print("\n训练中...")
rf_model.fit(X_train, y_reg_train)

# 预测
y_reg_pred = rf_model.predict(X_test)

# 评估
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"\n回归评估结果:")
print(f"R² 分数: {r2:.4f}")
print(f"RMSE: {rmse:.2f} 千卡")
print(f"MSE: {mse:.2f}")

# 交叉验证（5折）
cv_scores = cross_val_score(rf_model, X, y_reg, cv=5, scoring='r2')
print(f"\n5折交叉验证 R² 分数: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 特征重要性
feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(f"\n特征重要性:")
print(feature_importance)

# 保存模型
joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest_regressor.pkl"))
print(f"\n模型已保存: {os.path.join(MODEL_DIR, 'random_forest_regressor.pkl')}")

# ============================================================
# 模型2：SVM分类器（预测蛋白质等级）
# ============================================================
print("\n" + "=" * 60)
print("模型2：SVM分类器（预测蛋白质等级）")
print("=" * 60)

# 对特征进行标准化（SVM对尺度敏感）
scaler = StandardScaler()

# 重新划分训练集和测试集，确保y_clf与X对应
X_train_clf, X_test_clf, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# 训练SVM（RBF核，使用类别权重平衡）
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',  # 自动处理类别不平衡
    random_state=42
)

print("\n训练中...")
svm_model.fit(X_train_clf_scaled, y_clf_train)

# 预测
y_clf_pred = svm_model.predict(X_test_clf_scaled)

# 评估
accuracy = accuracy_score(y_clf_test, y_clf_pred)
print(f"\n分类评估结果:")
print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n分类报告:")
print(classification_report(y_clf_test, y_clf_pred))

# 交叉验证
cv_scores_clf = cross_val_score(svm_model, X_train_clf_scaled, y_clf_train, cv=5, scoring='accuracy')
print(f"\n5折交叉验证准确率: {cv_scores_clf.mean():.4f} (±{cv_scores_clf.std():.4f})")

# 保存模型
joblib.dump(svm_model, os.path.join(MODEL_DIR, "svm_classifier.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"\n模型已保存: {os.path.join(MODEL_DIR, 'svm_classifier.pkl')}")
print(f"标准化器已保存: {os.path.join(MODEL_DIR, 'scaler.pkl')}")

# ============================================================
# 可视化（可选）
# ============================================================
print("\n" + "=" * 60)
print("生成可视化图表...")
print("=" * 60)

# 图1：特征重要性
plt.figure(figsize=(8, 5))
plt.barh(feature_importance['特征'], feature_importance['重要性'], color='skyblue')
plt.xlabel('重要性')
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'), dpi=150)
print(f"已保存: feature_importance.png")

# 图2：混淆矩阵
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_clf_test, y_clf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('SVM分类混淆矩阵')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150)
print(f"已保存: confusion_matrix.png")

print("\n" + "=" * 60)
print("模型训练完成！")
print(f"模型文件保存在: {MODEL_DIR}/")
print("=" * 60)
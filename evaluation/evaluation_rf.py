import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# 使用Python引擎加载数据，跳过坏行
data = pd.read_csv('face_keypoints_test.csv', engine='python', on_bad_lines='skip')

# 显示数据集的基本信息
print(f'数据集共有 {data.shape[0]} 行和 {data.shape[1]} 列')
print(data.head())

# 检查是否存在缺失值并处理
if data.isnull().values.any():
    print("数据中存在缺失值，处理缺失数据...")
    data = data.dropna()  # 或者使用 data.fillna(method='ffill') 进行填充

# 提取标签和特征
y_true = data['label']
X = data.drop('label', axis=1).values  # 转换为模型输入所需的 numpy 数组

# 加载模型和标签编码器
label_encoder_path = os.path.join('ml_model', 'label_encoder.pkl')
rf_model_path = os.path.join('ml_model', 'rf_model.pkl')

label_encoder = joblib.load(label_encoder_path)
rf_model = joblib.load(rf_model_path)

# 对真实标签进行编码
y_true_encoded = label_encoder.transform(y_true)

# 使用模型进行预测
y_pred_encoded = rf_model.predict(X)

# 计算准确率和 F1 分数
accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
print(f'模型准确率: {accuracy*100:.2f}%')

f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
print(f'F1 分数: {f1:.2f}')

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_true_encoded, y_pred_encoded)
print("混淆矩阵:")
print(conf_matrix)

# 绘制混淆矩阵热图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# 使用交叉验证绘制准确率图表（适用于随机森林模型）
cv_scores = cross_val_score(rf_model, X, y_true_encoded, cv=5, scoring='accuracy')

plt.figure()
plt.plot(range(1, 6), cv_scores, marker='o')
plt.xlabel('Cross-Validation Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy for Random Forest Model')
plt.show()

# 数据探索：类别分布图
plt.figure()
sns.countplot(x=y_true)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# 数据探索：关键点可视化（为简洁，仅显示前5个关键点的成对图）
sns.pairplot(data.iloc[:, :5])  # 仅可视化前5列，便于阅读
plt.show()

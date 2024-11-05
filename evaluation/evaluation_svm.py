import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 尝试使用Python引擎读取CSV，并跳过坏行
data = pd.read_csv('face_keypoints_test.csv', engine='python', on_bad_lines='skip')

# 检查数据是否正确读取
print(f'数据共有 {data.shape[0]} 行和 {data.shape[1]} 列')
print(data.head())

# 检查是否存在缺失值
if data.isnull().values.any():
    print("数据中存在缺失值，进行处理...")
    data = data.dropna()  # 或者选择填充缺失值，如 data.fillna(method='ffill')

# 提取标签和特征
y_true = data['label']
X = data.drop('label', axis=1).values  # 转换为 numpy 数组

# 加载模型、标签编码器和标准化器
label_encoder_path = os.path.join('ml_model', 'label_encoder.pkl')
svm_model_path = os.path.join('ml_model', 'svm_model.pkl')
scaler_path = os.path.join('ml_model', 'scaler.pkl')

label_encoder = joblib.load(label_encoder_path)
svm_model = joblib.load(svm_model_path)
scaler = joblib.load(scaler_path)

# 对真实标签进行编码
y_true_encoded = label_encoder.transform(y_true)

# 对特征进行标准化
X_scaled = scaler.transform(X)

# 使用模型进行预测
y_pred_encoded = svm_model.predict(X_scaled)

# 计算准确率
accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
print(f'模型预测的准确率为: {accuracy*100:.2f}%')

# 计算 F1 Score
f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')  # 'weighted' 适用于不平衡数据集
print(f'F1 Score: {f1:.2f}')

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_true_encoded, y_pred_encoded)
print("Confusion Matrix:")
print(conf_matrix)
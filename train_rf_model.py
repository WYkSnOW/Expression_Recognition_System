import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# 加载数据集并处理，跳过有问题的行
def load_data(csv_file):
    # 使用 pandas 读取 CSV 文件，并跳过有问题的行
    try:
        data = pd.read_csv(csv_file, on_bad_lines='skip')
    except Exception as e:
        print(f"读取 CSV 文件时发生错误: {e}")
        return None, None, None
    
    # 提取标签列（label）和特征列（面部关键点坐标）
    try:
        labels = data['label']  # 提取标签列
        features = data.drop(columns=['label'])  # 去掉标签列的所有列作为特征
    except KeyError as e:
        print(f"CSV 文件中缺少 'label' 列或关键点数据: {e}")
        return None, None, None
    
    # 将标签进行编码（如果是文本标签）
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    return features, labels, label_encoder

# 训练随机森林模型
def train_rf_model(features, labels, model_output_path):
    if features is None or labels is None:
        print("特征或标签数据有问题，无法训练模型。")
        return

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 初始化随机森林分类器
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练模型
    rf.fit(X_train, y_train)

    # 预测测试集
    y_pred = rf.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    # 保存训练好的模型
    joblib.dump(rf, model_output_path)
    print(f"模型已保存到: {model_output_path}")

if __name__ == "__main__":
    # CSV 文件路径
    csv_file = "face_keypoints.csv"
    
    # 模型保存路径
    model_output_path = "rf_model.pkl"
    
    # 加载数据
    features, labels, label_encoder = load_data(csv_file)
    
    # 训练模型
    train_rf_model(features, labels, model_output_path)

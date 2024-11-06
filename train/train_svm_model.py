import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_data(csv_file):
    """
    加载 CSV 文件并准备数据
    """
    data = pd.read_csv(csv_file, on_bad_lines='skip')

    print(f"数据集共有 {len(data)} 行")
    print(data.head())

    if data.isnull().values.any():
        print("数据中存在缺失值，正在删除含有缺失值的行...")
        data = data.dropna()
        print(f"删除缺失值后，数据集共有 {len(data)} 行")

    if 'label' not in data.columns:
        print("错误：数据中未找到 'label' 列。")
        return None, None, None, None

    X = data.drop('label', axis=1).values
    y = data['label'].values

    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"标签映射：{label_mapping}")

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, label_encoder, scaler

def train_svm_model(X, y):
    """
    训练 SVM 模型
    """
    if len(X) == 0 or len(y) == 0:
        print("错误：数据量不足，无法训练模型。")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM 模型准确率: {accuracy * 100:.2f}%")

    return svm_classifier

def save_model(model, model_filename):
    """
    保存模型
    """
    if model is not None:
        joblib.dump(model, model_filename)
        print(f"模型已保存到 {model_filename}")
    else:
        print("模型为空，无法保存。")

def main():
    csv_file = "face_keypoints.csv"
    model_filename = "ml_model/svm_model.pkl"

    X, y, label_encoder, scaler = load_data(csv_file)

    if X is None or y is None:
        print("数据加载失败，无法继续。")
        return

    print(f"特征矩阵形状：{X.shape}")
    print(f"标签数组形状：{y.shape}")

    svm_model = train_svm_model(X, y)

    save_model(svm_model, model_filename)

    joblib.dump(label_encoder, 'ml_model/label_encoder.pkl')
    joblib.dump(scaler, 'ml_model/scaler.pkl')
    print("标签编码器已保存到 'ml_model/scaler.pkl")
    print("标准化器已保存到 ml_model/scaler.pkl")

if __name__ == "__main__":
    main()

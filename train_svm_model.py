import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


def load_data(csv_file):
    """
    加载 CSV 文件并准备数据
    :param csv_file: CSV 文件路径
    :return: 特征（X）和标签（y）
    """
    # 读取 CSV 文件
    data = pd.read_csv(csv_file)

    # 提取特征（所有的 x 和 y 坐标）和标签
    X = data.iloc[:, 1:].values  # 所有特征（x 和 y 坐标）
    y = data['label'].values  # 标签（类别）

    return X, y


def train_svm_model(X, y):
    """
    训练支持向量机（SVM）模型
    :param X: 特征数据
    :param y: 标签数据
    :return: 训练好的 SVM 模型
    """
    # 将数据拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建 SVM 分类器
    svm_classifier = SVC(kernel='linear')  # 你也可以尝试 'rbf' 或 'poly' 内核

    # 训练模型
    svm_classifier.fit(X_train, y_train)

    # 测试模型
    y_pred = svm_classifier.predict(X_test)

    # 输出模型准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")

    return svm_classifier


def save_model(model, model_filename):
    """
    保存训练好的模型到文件
    :param model: 训练好的模型
    :param model_filename: 模型保存的文件名
    """
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")


def main():
    # 定义 CSV 文件路径和模型保存路径
    csv_file = "face_keypoints.csv"
    model_filename = "svm_model.pkl"

    # 加载数据
    X, y = load_data(csv_file)

    # 训练 SVM 模型
    svm_model = train_svm_model(X, y)

    # 保存训练好的模型
    save_model(svm_model, model_filename)


if __name__ == "__main__":
    main()

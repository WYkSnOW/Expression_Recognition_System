import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_data(csv_file):
    """
    Load CSV file and prepare data
    """
    data = pd.read_csv(csv_file, on_bad_lines='skip')

    print(f"Data sets share {len(data)} OK")
    print(data.head())

    if data.isnull().values.any():
        print("There are missing values ​​in the data，Deleting rows with missing values...")
        data = data.dropna()
        print(f"After removing missing values，Data sets share {len(data)} OK")

    if 'label' not in data.columns:
        print("mistake：Not found in data 'label' List。")
        return None, None, None, None

    X = data.drop('label', axis=1).values
    y = data['label'].values

    # Tag encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"tag mapping：{label_mapping}")

    # Feature normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, label_encoder, scaler

def train_svm_model(X, y):
    """
    Train an SVM model
    """
    if len(X) == 0 or len(y) == 0:
        print("mistake：Not enough data，Unable to train model。")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM model accuracy: {accuracy * 100:.2f}%")

    return svm_classifier

def save_model(model, model_filename):
    """
    Save model
    """
    if model is not None:
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")
    else:
        print("Model is empty，Unable to save。")

def main():
    csv_file = "face_keypoints.csv"
    model_filename = "ml_model/svm_model.pkl"

    X, y, label_encoder, scaler = load_data(csv_file)

    if X is None or y is None:
        print("Data loading failed，Unable to continue。")
        return

    print(f"Feature matrix shape：{X.shape}")
    print(f"label array shape：{y.shape}")

    svm_model = train_svm_model(X, y)

    save_model(svm_model, model_filename)

    joblib.dump(label_encoder, 'ml_model/label_encoder.pkl')
    joblib.dump(scaler, 'ml_model/scaler.pkl')
    print("Label encoder saved to 'ml_model/scaler.pkl")
    print("Normalizer saved to ml_model/scaler.pkl")

if __name__ == "__main__":
    main()

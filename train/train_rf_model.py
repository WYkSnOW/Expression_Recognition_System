import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset and process it, skipping problematic rows
def load_data(csv_file):
    # Read CSV file using pandas and skip problematic lines
    try:
        data = pd.read_csv(csv_file, on_bad_lines='skip')
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None, None, None
    
    # Extract label column (label) and feature column (facial key point coordinates)
    try:
        labels = data['label']  # Extract label column
        features = data.drop(columns=['label'])  # Remove all columns of label column as features
    except KeyError as e:
        print(f"'label' column or keypoint data is missing from CSV file: {e}")
        return None, None, None
    
    # Encode the label (if it is a text label)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    return features, labels, label_encoder

# Train a random forest model
def train_rf_model(features, labels, model_output_path):
    if features is None or labels is None:
        print("There is a problem with the feature or label data，Unable to train model。")
        return

    # Divide training set and test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize the random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Training model
    rf.fit(X_train, y_train)

    # Predict test set
    y_pred = rf.predict(X_test)

    # Evaluation model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(rf, model_output_path)
    print(f"Model saved to: {model_output_path}")

if __name__ == "__main__":
    # CSV file path
    csv_file = "face_keypoints.csv"
    
    # Model save path
    model_output_path = "ml_model/rf_model.pkl"
    
    # Load data
    features, labels, label_encoder = load_data(csv_file)
    
    # Training model
    train_rf_model(features, labels, model_output_path)

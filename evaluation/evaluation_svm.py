import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file using Python engine, skipping bad lines
data = pd.read_csv('face_keypoints_test.csv', engine='python', on_bad_lines='skip')

# Display basic data info
print(f'Dataset contains {data.shape[0]} rows and {data.shape[1]} columns')
print(data.head())

# Check for missing values and handle them
if data.isnull().values.any():
    print("Missing values detected, handling...")
    data = data.dropna()  # Or use data.fillna(method='ffill')

# Extract labels and features
y_true = data['label']
X = data.drop('label', axis=1).values  # Convert to numpy array

# Load model, label encoder, and scaler
label_encoder_path = os.path.join('ml_model/svm', 'label_encoder.pkl')
svm_model_path = os.path.join('ml_model/svm', 'svm_model.pkl')
scaler_path = os.path.join('ml_model/svm', 'scaler.pkl')

label_encoder = joblib.load(label_encoder_path)
svm_model = joblib.load(svm_model_path)
scaler = joblib.load(scaler_path)

# Encode true labels
y_true_encoded = label_encoder.transform(y_true)

# Scale features
X_scaled = scaler.transform(X)

# Predict results
y_pred_encoded = svm_model.predict(X_scaled)

# Calculate accuracy and F1 Score
accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
print(f'Model Accuracy: {accuracy*100:.2f}%')

f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
print(f'F1 Score: {f1:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_encoded, y_pred_encoded)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Cross-Validation Accuracy Plot
cv_scores = cross_val_score(svm_model, X_scaled, y_true_encoded, cv=5, scoring='accuracy')

plt.figure()
plt.plot(range(1, 6), cv_scores, marker='o')
plt.xlabel('Cross-Validation Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy of SVM Model')
plt.show()

# Data Exploration: Class Distribution
plt.figure()
sns.countplot(x=y_true)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# Data Exploration: Pair Plot of Features (First 5 Keypoints)
sns.pairplot(data.iloc[:, :5])
plt.suptitle('Pair Plot of First 5 Keypoints', y=1.02)
plt.show()

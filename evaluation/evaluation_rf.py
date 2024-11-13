import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data using Python engine, skip bad lines
data = pd.read_csv('face_keypoints_test.csv', engine='python', on_bad_lines='skip')

# Display basic information of the dataset
print(f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns')
print(data.head())

# Check for missing values and handle them
if data.isnull().values.any():
    print("Missing values detected in the data, handling missing data...")
    data = data.dropna()  # Alternatively, use data.fillna(method='ffill') to fill

# Extract labels and features
y_true = data['label']
X = data.drop('label', axis=1).values  # Convert to numpy array required for model input

# Load model and label encoder
label_encoder_path = os.path.join('ml_model/svm', 'label_encoder.pkl')
rf_model_path = os.path.join('ml_model/rf', 'rf_model.pkl')

label_encoder = joblib.load(label_encoder_path)
rf_model = joblib.load(rf_model_path)

# Encode the true labels
y_true_encoded = label_encoder.transform(y_true)

# Make predictions using the model
y_pred_encoded = rf_model.predict(X)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
print(f'Model Accuracy: {accuracy*100:.2f}%')

f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
print(f'F1 Score: {f1:.2f}')

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true_encoded, y_pred_encoded)
print("Confusion Matrix:")
print(conf_matrix)

# Plot heatmap of confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Plot accuracy chart using cross-validation (suitable for random forest model)
cv_scores = cross_val_score(rf_model, X, y_true_encoded, cv=5, scoring='accuracy')

plt.figure()
plt.plot(range(1, 6), cv_scores, marker='o')
plt.xlabel('Cross-Validation Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy for Random Forest Model')
plt.show()

# Data exploration: Class distribution chart
plt.figure()
sns.countplot(x=y_true)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# Data exploration: Keypoint visualization (for simplicity, only shows pairplot of the first 5 keypoints)
sns.pairplot(data.iloc[:, :5])  # Visualize only the first 5 columns for readability
plt.show()

import os
import joblib

# Define the path of label_encoder
label_encoder_path = os.path.join('ml_model', 'label_encoder.pkl')

# Load label_encoder
label_encoder = joblib.load(label_encoder_path)

# Output the content of label_encoder
print("Label Encoder Classes:")
print(label_encoder.classes_)

# Show the encoding corresponding to each tag
for index, label in enumerate(label_encoder.classes_):
    print(f"Label '{label}' is encoded as {index}")

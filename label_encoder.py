import os
import joblib

# 定义label_encoder的路径
label_encoder_path = os.path.join('ml_model', 'label_encoder.pkl')

# 加载label_encoder
label_encoder = joblib.load(label_encoder_path)

# 输出label_encoder的内容
print("Label Encoder Classes:")
print(label_encoder.classes_)

# 显示每个标签对应的编码
for index, label in enumerate(label_encoder.classes_):
    print(f"Label '{label}' is encoded as {index}")

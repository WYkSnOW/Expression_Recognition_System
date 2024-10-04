import streamlit as st
import pandas as pd

#
st.title('Facial Expression Recognition System Using Machine Learning')

#
st.text('By: Waiyuk Kwong, Zhihui Chen, Tyler Lin, Blane R. York, and Carter D. Robinson')

#
st.header('Introduction/Background')

st.write("""
Facial expressions are a key aspect of human communication, conveying emotions and intentions without the need for words. 
As one of the most dynamic features of the human body, they provide critical information about a person's emotional state [2]. 
Recognizing these expressions through machine learning has applications in areas such as human-computer interaction, research, education, and mental healthcare.
""")

st.subheader('Literature Overview')
st.markdown("""
- **Paper 1 (Huang & Samonte, 2024)**: Explores the use of **Google MediaPipe** to track facial key points, combined with emotion analysis and other factors like eye movement, to assess engagement.
""")
st.markdown("""
- **Paper 2 (Sadikoğlu & Mohamed, 2022)**: Focuses on **Convolutional Neural Networks (CNNs)** and their ability to recognize facial expressions by extracting features from images. It highlights the advantages of CNNs and discusses the use of transfer learning.
""")
st.markdown("""
- **Paper 3 (Wang, Li, & Zhao, 2010)**: Describes the use of **Active Shape Models (ASM)** and **Support Vector Machines (SVM)** for real-time recognition, emphasizing geometric features for expression recognition.
""")

st.subheader('Dataset Description')
st.write("""
We will use the FER-2013 dataset, which contains over 35,000 labeled images across emotion categories like anger, fear, happiness, and sadness, to implement a model that can detect expressions in real time. 
Key tools include Google MediaPipe for extracting facial key points and Convolutional Neural Networks (CNNs) for feature extraction, along with additional machine learning algorithms to enhance accuracy and efficiency.
""")

st.subheader("Dataset Link")
st.markdown("""
You can access the FER-2013 dataset from the following link:
[FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
""")

#
st.header('Problem Definition')
st.markdown("""
**Problem:**

- How can facial expressions be accurately recognized in real-time scenarios using facial detection systems?

**Motivation:**

- While current facial expression recognition systems are effective, they often struggle with real-time performance and robustness. By combining facial key points and image data, we aim to develop a more efficient and robust system.
""")

#
st.header('Methods')

st.subheader('Data Preprocessing Methods')
st.markdown("""
1. **Facial Key Point Extraction**: Using **Google MediaPipe**, we will extract 468 facial key points from each image, capturing geometric changes that reflect subtle expressions [1].
2. **Image Normalization**: We will standardize image pixel values to reduce the effect of lighting inconsistencies and contrast variations, ensuring uniform input for the model.
3. **Data Augmentation**: Techniques like rotation, flipping, and cropping will be used to artificially expand the dataset and improve the model's ability to generalize across different facial angles and expressions.
""")

st.subheader('Machine Learning Algorithms/Models')
st.markdown("""
1. **Convolutional Neural Network (CNN)**: A deep learning model will be used to automatically extract features from the input images and classify expressions based on the extracted patterns.
2. **Support Vector Machine (SVM)**: This model will use the facial key points extracted from MediaPipe to classify facial expressions based on geometric features.
3. **Random Forest**: We will use an ensemble learning model that combines both image-based and key point-based features to improve classification accuracy. Random Forests are known for their robustness in handling complex, multi-modal datasets.
""")

#
st.header('Results and Discussion')

st.subheader('Quantitative Metrics')
st.markdown("""
- **Accuracy**: The percentage of correctly predicted expressions out of the total number of predictions.
- **F1-Score**: A balanced metric that combines precision and recall, providing better insight when dealing with imbalanced expression categories.
- **Confusion Matrix**: Evaluates how well the model distinguishes between different facial expressions, helping to identify misclassifications.
""")

st.subheader('Project Goals')
st.markdown("""
The goal of the project is to develop a facial expression recognition model with high accuracy in real-time scenarios, ensuring robustness across varying lighting conditions and angles.
An ethical consideration is to ensure that the model does not misinterpret expressions across different ethnicities or genders.
""")

st.subheader('Expected Results')
st.write('The model will be capable of accurately detecting and classifying multiple facial expressions and providing real-time feedback in real-world applications.')

#
st.header('References')
st.markdown("""
1. A. Huang and M. J. C. Samonte, "Using Emotion Analysis, Eye tracking, and Head Movement to Monitor Student Engagement among ESL Students with Facial Recognition Algorithm (Mediapipe)," 2024 7th International Conference on Advanced Algorithms and Control Engineering (ICAACE), Shanghai, China, 2024, pp. 509-513, doi: [10.1109/ICAACE61206.2024.10548871](https://doi.org/10.1109/ICAACE61206.2024.10548871).
2. F. M. Sadikoğlu and M. Idle Mohamed, "Facial Expression Recognition Using CNN," 2022 International Conference on Artificial Intelligence in Everything (AIE), Lefkosa, Cyprus, 2022, pp. 95-99, doi: [10.1109/AIE57029.2022.00025](https://doi.org/10.1109/AIE57029.2022.00025).
3. K. Wang, R. Li, and L. Zhao, "Real-time facial expressions recognition system for service robot based-on ASM and SVMs," 2010 8th World Congress on Intelligent Control and Automation, Jinan, 2010, pp. 6637-6641, doi: [10.1109/WCICA.2010.5554164](https://doi.org/10.1109/WCICA.2010.5554164).
""")

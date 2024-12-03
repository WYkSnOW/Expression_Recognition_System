import cv2
import streamlit as st
from UI.helper.emotion_model import predict_emotion
from UI.helper.utils import convert_frame_to_image
from UI.helper.face_detection import detect_faces, draw_face_boxes
from UI.helper.face_mesh import FaceMeshDetector
import time
from PIL import Image
import pandas as pd
import numpy as np 

st.set_page_config(layout="wide")

import streamlit as st
import pandas as pd
import numpy as np 

#
st.title('Facial Expression Recognition System Using Machine Learning')

#
st.text('By: Waiyuk Kwong, Zhihui Chen, Tyler Lin, Blane R. York, and Carter D. Robinson')

#
st.header('Introduction/Background', divider="gray")

st.write("""
Facial expressions are a key aspect of human communication, conveying emotions and intentions without the need for words. 
As one of the most dynamic features of the human body, they provide critical information about a person's emotional state [2]. 
Recognizing these expressions through machine learning has applications in areas such as human-computer interaction, research, and education.
""")

st.subheader('Literature Overview')
with st.expander('Literature Overview'):
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
""")
st.markdown("""
[FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
""")

#
st.header('Problem Definition', divider="gray")
st.markdown("""
**Problem:**

- How can facial expressions be accurately recognized in real-time scenarios using facial detection systems?

**Motivation:**

- While current facial expression recognition systems are effective, they often struggle with real-time performance and robustness. By combining facial key points and image data, we aim to develop a more efficient and robust system.
""")

#
st.header('Methods', divider="gray")

st.subheader('Data Preprocessing Methods')
st.markdown("""
1. **Facial Key Point Extraction**: Using **Google MediaPipe**, we will extract 468 facial key points from each image, capturing geometric changes that reflect subtle expressions [1].
            MediaPipe is efficient for real-time key point detection and reduces the complexity of manually coding key point extraction.  
            **Library**: `mediapipe`
2. **Image Normalization**: We will standardize image pixel values to reduce the effect of lighting inconsistencies and contrast variations, ensuring uniform input for the model.   
            **Library**: `cv2`
3. **Data Augmentation**: Techniques like rotation, flipping, and cropping will be used to artificially expand the dataset and improve the model's ability to generalize across different facial angles and expressions.  
            **Library**: `cv2` `numpy`
""")

st.subheader('Machine Learning Algorithms/Models')
st.markdown("""
1. **Random Forest**: Random Forest will use the facial landmark data to capture the unique patterns associated with different expressions using multiple decision trees. This approach provides a reliable model that’s interpretable and relatively fast to train and evaluate.
2. **Support Vector Machine (SVM)**: SVM will use facial landmark data to classify expressions by finding the optimal boundaries between different classes based on landmark positions. This approach allows for high precision in recognizing expression.
3. **Convolutional Neural Network (CNN)**:The CNN will classify facial expressions directly from raw image data, using multiple layers to capture specific visual features and patterns that indicate different expressions. This layered approach enables the CNN to identify detailed facial features, making it effective at accurately recognizing expressions.
""")

#
st.header('Results and Discussion', divider="gray")

st.subheader('Random Forest Model')
st.markdown("""
- **Accuracy**: 40.07%
- **F1-Score**: 0.37
- **Confusion Matrix**: 
""")
st.image("streamlit/rf_confusion_matrix.jpg", caption="Random Forest Confusion Matrix")

st.markdown("""
- **Cross-Validation Accuracy**: 
""")
st.image("streamlit/rf_cross_validation.jpg", caption="Cross-Validation Accuracy for Random Forest Model")
st.markdown("""
- **Class Distribution**: 
""")
st.image("streamlit/rf_class_distribution.jpg", caption="Class Distribution for Random Forest Model")

st.subheader('Random Forest Analysis')
st.write("""
The Random Forest model shows limited performance with an accuracy of 40.07% and an F1-score of 0.37.
 This indicates that the model struggles with generalization and class prediction. The confusion matrix and class distribution table show a strong bias toward predicting "happy" and a consistent failure to accurately predict "disgust." 
 Additionally, the cross-validation accuracy plot shows that the model's performance is inconsistent across different data subsets, meaning its accuracy varies depending on the subset of data used for testing.
""")

st.subheader('Support Vector Machine (SVM)')
st.markdown("""
- **Accuracy**: 50.28%
- **F1-Score**: 0.48
- **Confusion Matrix**: 
""")
st.image("streamlit/svm_confusion_matrix.jpg", caption="SVM Confusion Matrix")

st.markdown("""
- **Cross-Validation Accuracy**: 
""")
st.image("streamlit/svm_cross_validation.jpg", caption="Cross-Validation Accuracy for SVM")
st.markdown("""
- **Class Distribution**: 
""")
st.image("streamlit/svm_class_distribution.jpg", caption="Class Distribution for SVM")


st.subheader('Support Vector Machine Analysis')
st.write("""
The Support Vector Machine Model ahieved an accuracy of 50.28% and F1-score of 0.48. The model performed better than the Random Forest Mode, 
but still struggles to classify facial expression acccurately. Similar to the Random Forest model, the SVM model is also biased toward predicting "happy" and misclassifyling other expressions. 
Although the overall accruacy in cross validation accuracy plot is around 50%, the variability across folds shows that the SVM model's performance also depends on the specific data subset used for training and testing.
""")

st.subheader('Convolutional Neural Network (CNN)')
st.markdown("""
- **Accuracy**:  64.29%
- **F1-Score**: 0.6378
- **Confusion Matrix**: 
""")
st.image("streamlit/cnn_confusion_matrix.jpg", caption="CNN Confusion Matrix")

st.markdown("""
- **Training and Validation Accuracy**: 
""")
st.image("streamlit/cnn_train_validation_accuracy.jpg", caption="Training and Validation Accuracy for CNN")
st.markdown("""
- **Training and Validation Loss**: 
""")
st.image("streamlit/cnn_train_validation_loss.jpg", caption="Training and Validation Loss for CNN")


st.subheader('Convolutional Neural Network Analysis')
st.write("""
The Convolutional Neural Network model shows a significant improvement with an accuracy of 64.29% and a F1-Score of 0.6378.
The confusion matrix shows that CNN has better predictions with a higher accuracy in the happy and neutral classes. 
Although misclassifications still remain in the classes such as fear, sad, and angry, there are much less than the previous 2 models. 
The training and validation accuracy curve indicates that the model is improving steadily until 65-70% where the curve flattens out. 
Similarly, the loss curve shows consistent decrease in both training and validation loss, indicating that the model is learning effectively on the training and validation data. 
Therefore, CNN performs much better than the previous models by effectively capturing patterns in the data.
""")

st.subheader('Overall Analysis')
st.write("""
Of the three models, the CNN model demonstrates the strongest performance, obtaining the highest accuracy and F1-score, while the Random Forest and SVM models show weaker results.
A key issue across all models is the significant misclassification of certain emotions. This may be due to that fact that emotions such as fear, disgust, and angry share similar features.  Another factor that may cause the misclassification is 
the imbalance in classes. Some emotions such as happy has a larger dataset than disgust. This imbalance may lead to bias towards the majority class,
making it harder for the models to accurately predict minority classes. 
Additionally, the 48x48 pixel size of the images may have limited the model's ability to capture fine details, constraining the amount of available visual information that could differentiate similar emotions.
""")

st.subheader('Next Steps')
st.write("""
After the analysis of our 3 models, it seems that our accuracy is still on the lower end. 
We will attempt to improve our model by improving our data before we retrain them. 
Also, we want to make a user interface where users can insert images into our model and output its prediction on the emotion.
""")

# Initialize state variables
if "mode" not in st.session_state:
    st.session_state.mode = "IMAGE"
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "calculated_label_name" not in st.session_state:
    st.session_state.calculated_label_name = "No Face Detected"
if "loading_camera" not in st.session_state:
    st.session_state.loading_camera = False
if "camera" not in st.session_state:
    st.session_state.camera = None 

# Mode Buttons
col1, col2, col3 = st.columns([15, 1, 1])
col1_placeholder = col1.empty()
with col2:
    if st.button("Image", key="image_button", help="Switch to Image Mode"):
        st.session_state.mode = "IMAGE"
        st.session_state.camera_active = False

with col3:
    if st.button("Live", key="live_button", help="Switch to Live Mode"):
        st.session_state.mode = "LIVE"

# Image Mode
if st.session_state.mode == "IMAGE":
    st.title("Upload Your Image")
    upload_text = st.empty()
    uploaded_file = st.file_uploader(f"", type=["jpg", "jpeg", "png"], label_visibility="hidden")
    if uploaded_file: 
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state.upload_text = "Re-Upload an Image"
        label = predict_emotion(image)
        st.session_state.calculated_label_name = label
        with upload_text:
            st.markdown("Re Upload Image")
        with col1_placeholder:
            st.markdown(f"**Result:** `{st.session_state.calculated_label_name}`")
    else:
        st.session_state.calculated_label_name = "No Face Detected"
        with col1_placeholder:
            st.markdown(f"**Result:** `{st.session_state.calculated_label_name}`")
        with upload_text:
            st.markdown("Upload Image")
# Live Mode
elif st.session_state.mode == "LIVE":
    st.session_state.calculated_label_name = "No Face Detected"
    st.title("Live Video")
    
    if not st.session_state.camera_active and not st.session_state.loading_camera:
        st.session_state.calculated_label_name = "No Face Detected"
        st.session_state.camera = None
        # Show Open Camera button if the camera is inactive
        if st.button("Open Camera", key="open_camera_button"):
            st.session_state.loading_camera = True
            st.rerun()  
    elif st.session_state.loading_camera:
        st.write("Loading Camera...")
        # Try to open camera feed
        st.session_state.camera = cv2.VideoCapture(0)
        if not st.session_state.camera.isOpened():
            st.error("Error: Could not access the camera.")
            st.session_state.loading_camera = False
            st.session_state.camera_active = False
            del st.session_state.camera
        else:
            st.session_state.loading_camera = False
            st.session_state.camera_active = True
            st.rerun()  # Rerun to show the video and Close Camera button
    elif st.session_state.camera_active:
        # Show Close Camera button and video
        if st.button("Close Camera", key="close_camera_button"):
            st.session_state.camera_active = False
            st.session_state.camera.release()  
            del st.session_state.camera  
            st.rerun() # Rerun to close video
        video_placeholder = st.empty()  # Placeholder for video
        result_placeholder = st.empty()  # Placeholder for the result, placed below the video
        while st.session_state.camera is not None and st.session_state.camera.isOpened():
                detector = FaceMeshDetector()
                
                # Read a single frame from the camera
                ret, frame = st.session_state.camera.read()
                if not ret:
                    st.error("Error: Failed to capture frame.")
                    st.session_state.camera_active = False
                    st.session_state.camera.release()
                    del st.session_state.camera
                    st.rerun()
                else:
                    # Face detection
                    face_boxes = detect_faces(frame)

                    # Draw face boxes
                    if face_boxes:
                        draw_face_boxes(frame, face_boxes)

                    # Apply FaceMesh
                    frame, faces_mesh, face_detected = detector.find_face_mesh(frame, draw=True)

                    # Predict Emotion
                    if face_detected and face_boxes:
                        x, y, x_end, y_end = face_boxes[0]
                        face_image = convert_frame_to_image(frame, x, y, x_end, y_end)
                        st.session_state.calculated_label_name = predict_emotion(face_image)
                    else:
                        st.session_state.calculated_label_name = "No Face Detected"

                    # Convert frame to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Update video feed in placeholder
                    video_placeholder.image(frame_rgb, channels="RGB")
                    
                    with result_placeholder:
                        st.markdown(f"**Result:** `{st.session_state.calculated_label_name}`")



#
st.header('References', divider="gray")
with st.expander('References'):
    st.markdown("""
    1. A. Huang and M. J. C. Samonte, "Using Emotion Analysis, Eye tracking, and Head Movement to Monitor Student Engagement among ESL Students with Facial Recognition Algorithm (Mediapipe)," 2024 7th International Conference on Advanced Algorithms and Control Engineering (ICAACE), Shanghai, China, 2024, pp. 509-513, doi: [10.1109/ICAACE61206.2024.10548871](https://doi.org/10.1109/ICAACE61206.2024.10548871).
    2. F. M. Sadikoğlu and M. Idle Mohamed, "Facial Expression Recognition Using CNN," 2022 International Conference on Artificial Intelligence in Everything (AIE), Lefkosa, Cyprus, 2022, pp. 95-99, doi: [10.1109/AIE57029.2022.00025](https://doi.org/10.1109/AIE57029.2022.00025).
    3. K. Wang, R. Li, and L. Zhao, "Real-time facial expressions recognition system for service robot based-on ASM and SVMs," 2010 8th World Congress on Intelligent Control and Automation, Jinan, 2010, pp. 6637-6641, doi: [10.1109/WCICA.2010.5554164](https://doi.org/10.1109/WCICA.2010.5554164).
    """)


data = {
    "Name": ["Waikyuk Kwong", "Zhihui Chen", "Tyler Lin", "Blane R. York", "Carter D. Robinson"],
    "Proposal Contributions": [
        "Data Preprocessing, Implement Models, Visualization", 
        "Track Progress, Evaluate Models, Write report",
        "Write Report, Streamlit, Evaluate Confusion Matrix",
        "Discuss, Evaluate Accuracy",
        "Discuss, Evaluate F1 score"]
}

df = pd.DataFrame(data)
df = df.set_index("Name")
st.subheader("Contribution Table")
st.dataframe(df)


st.subheader("Gantt Chart", divider="gray")
st.image("streamlit/Gantt.jpg", caption="Gantt Chart")
st.markdown("""
[Gantt Chart](https://docs.google.com/spreadsheets/d/16sWj1XushsbAo5WwqrAq6MPuiGra0VFZrKK61rONgeo/edit?usp=sharing)
""")

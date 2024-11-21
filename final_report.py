import cv2
import streamlit as st
from UI.helper.emotion_model import predict_emotion
from UI.helper.utils import convert_frame_to_image
from UI.helper.face_detection import detect_faces, draw_face_boxes
from UI.helper.face_mesh import FaceMeshDetector
import time
from PIL import Image
st.set_page_config(layout="wide")

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
col1, col2, col3 = st.columns([20, 1, 1])
col1_placeholder = col1.empty()
with col1_placeholder:
    st.markdown(f"**Result:** `{st.session_state.calculated_label_name}`")
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
        video_placeholder = st.empty()  
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

                    # Add a small delay for frame rate control
                    time.sleep(0.1)
                    
                #Update Emotion Text
                with col1_placeholder:
                    st.markdown(f"**Result:** `{st.session_state.calculated_label_name}`")

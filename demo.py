import streamlit as st
import cv2
import time
from PIL import Image
from io import BytesIO
import base64
from UI.helper.face_detection import detect_faces, draw_face_boxes
from UI.helper.emotion_model import predict_emotion
from UI.helper.utils import convert_frame_to_image
from UI.helper.face_mesh import FaceMeshDetector



def display_picture_mode():
    enable_camera = st.checkbox("Enable Camera")
    picture = st.camera_input("Take a picture", disabled=not enable_camera)
    if picture:
        st.image(picture, caption="Captured Image", use_column_width=True)
        img = Image.open(picture)
        label = predict_emotion(img)
        st.markdown(f"**Result:** `{label}`")
        return 
    st.markdown("### Or upload an image:")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="visible")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

def run_live_mode():
    # Initialize session state for camera status
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False  # Camera is initially off

    # Define CSS styles
    st.markdown(
        """
        <style>
        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 720px;
            height: 405px;
            margin: 20px auto;
            border: 5px solid black; /* Black border */
            border-radius: 10px;
            overflow: hidden;
            background-color: white; /* White background */
            position: relative;
        }
        .loading-text {
            position: absolute;
            color: black;
            font-size: 24px;
            font-weight: bold;
        }
        .video-container img {
            object-fit: cover;
            width: 720px;
            height: 405px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Button for starting or closing the camera
    toggle_camera = st.button("Start/Close Camera")

    # Handle camera toggle logic
    if toggle_camera:
        st.session_state.camera_active = not st.session_state.camera_active

    # Create an empty placeholder for displaying the video
    placeholder = st.empty()
    
    result_placeholder = st.empty()

    # Initialize FaceMeshDetector
    detector = FaceMeshDetector()

    # If camera is active
    if st.session_state.camera_active:
        # Show loading initially
        placeholder.markdown(
            '''
            <div class="video-container">
                <span class="loading-text">Loading...</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        # Open the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open the camera.")
            st.session_state.camera_active = False  # Reset camera state
            return

        # Start the camera loop
        loading_displayed = True
        while st.session_state.camera_active:
            ret, frame = cap.read()

            if not ret:
                # Keep displaying "Loading..." if no frame is captured
                if not loading_displayed:
                    placeholder.markdown(
                        '''
                        <div class="video-container">
                            <span class="loading-text">Loading...</span>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )
                    loading_displayed = True
                time.sleep(0.03)
                continue

            # Successfully captured a frame; remove "Loading..." message
            loading_displayed = False

            # Process the frame (FaceMesh and Emotion Detection)
            frame, faces_mesh, face_detected = detector.find_face_mesh(frame, draw=True)
            face_boxes = detect_faces(frame)
            draw_face_boxes(frame, face_boxes)

            current_label = "No Face Detected"
            
            if face_detected and face_boxes:
                x, y, x_end, y_end = face_boxes[0]  # Only analyze the first detected face
                face_image = convert_frame_to_image(frame, x, y, x_end, y_end)
                current_label = predict_emotion(face_image)

            result_placeholder.markdown(f"**Result:** `{current_label}`")
            # Display the prediction on the video frame
            cv2.putText(frame, f"Prediction: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Resize the frame to 720x405 and convert to RGB format
            frame = cv2.resize(frame, (720, 405))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to JPEG and encode as Base64
            pil_img = Image.fromarray(frame)
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode()

            # Update the placeholder with the video frame
            html_code = f'''
                <div class="video-container">
                    <img src="data:image/jpeg;base64,{img_b64}" />
                </div>
            '''
            placeholder.markdown(html_code, unsafe_allow_html=True)

            # Add a short delay to reduce CPU usage
            time.sleep(0.03)

        # Release the camera when stopped
        cap.release()
    else:
        # If camera is not active, clear the placeholder
        placeholder.markdown(
            '''
            <div class="video-container">
                <!-- Placeholder for video -->
            </div>
            ''',
            unsafe_allow_html=True,
        )
        result_placeholder.markdown("**Result:** `No Camera Active`")

def main():
    st.title("FaceMesh and Emotion Recognition")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Live Mode"):
            st.session_state.mode = "LIVE"
    with col2:
        if st.button("Picture Mode"):
            st.session_state.mode = "PICTURE"

    if st.session_state.get("mode") == "LIVE":
        run_live_mode()
    elif st.session_state.get("mode") == "PICTURE":
        display_picture_mode()


if "st_page" in st.session_state or __name__ == "__main__":
    main()

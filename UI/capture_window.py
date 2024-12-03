import streamlit as st
import cv2
import time
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from helper.face_detection import detect_faces, draw_face_boxes
from helper.emotion_model import predict_emotion
from helper.utils import convert_frame_to_image
from helper.face_mesh import FaceMeshDetector

# Define the VideoTransformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = FaceMeshDetector()
        self.show_face_mesh = True
        self.show_face_box = True
        self.detection_interval = 1  # in seconds
        self.current_label = "No Face Detected"
        self.last_detection_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize frame to 720x405
        img = cv2.resize(img, (720, 405))

        # Detect face mesh
        img, faces_mesh, face_detected = self.detector.find_face_mesh(img, draw=self.show_face_mesh)

        # Always run detection logic
        face_boxes = detect_faces(img)

        # If needed, draw face boxes
        if self.show_face_box:
            draw_face_boxes(img, face_boxes)

        # Emotion prediction logic
        if face_detected and face_boxes:
            current_time = time.time()
            if current_time - self.last_detection_time >= self.detection_interval:
                x, y, x_end, y_end = face_boxes[0]  # Only process the first face
                face_image = convert_frame_to_image(img, x, y, x_end, y_end)
                self.current_label = predict_emotion(face_image)
                self.last_detection_time = current_time
        else:
            self.current_label = "No Face Detected"

        # Display prediction result
        cv2.putText(img, f"Predicted: {self.current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    st.title("Real-Time FaceMesh & Emotion Detection")

    # Create a 720x405 black frame
    black_frame = np.zeros((405, 720, 3), dtype=np.uint8)
    # Add a border to the black frame
    black_frame = cv2.copyMakeBorder(black_frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Display the black frame
    image_placeholder = st.image(black_frame, channels='BGR')

    # Create the start button
    start_button = st.button('Start')

    if start_button:
        # Remove the black frame
        image_placeholder.empty()

        # Start the video stream
        ctx = webrtc_streamer(
            key="face_mesh_emotion",
            video_transformer_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        # Add checkboxes for controlling FaceMesh and Face Box display
        if ctx.video_transformer:
            ctx.video_transformer.show_face_mesh = st.checkbox('Show FaceMesh', value=True)
            ctx.video_transformer.show_face_box = st.checkbox('Show Face Box', value=True)

if __name__ == "__main__":
    main()

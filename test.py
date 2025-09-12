# import streamlit as st
# import cv2
# from ultralytics import YOLO
# import tempfile
# import time

# # Load your YOLO model (replace with your model path)
# model = YOLO(rf"C:\Users\SRG Project\venv1\SFM\runs\detect\train11\weights\best.pt")  

# st.title("📹 Live Detection App")

# # Start/Stop button
# run = st.checkbox("Run on Camera Feed")

# # Streamlit placeholder for video
# frame_window = st.image([])

# # OpenCV video capture
# camera = cv2.VideoCapture(1)  # 0 = default webcam

# while run:
#     ret, frame = camera.read()
#     if not ret:
#         st.write("⚠️ Failed to grab frame")
#         break

#     # Run YOLO inference
#     results = model(frame)

#     # Annotate frame
#     annotated_frame = results[0].plot()

#     # Convert BGR (OpenCV) -> RGB (Streamlit expects RGB)
#     frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

#     # Show in streamlit
#     frame_window.image(frame_rgb)

#     # Small delay to prevent overload
#     time.sleep(0.02)

# camera.release()


import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2

# Load model
model = YOLO("best.pt")

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO
        results = model(img)
        annotated_img = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

st.title("📹 YOLO Live Detection (WebRTC)")
webrtc_streamer(key="yolo", video_processor_factory=YOLOProcessor)

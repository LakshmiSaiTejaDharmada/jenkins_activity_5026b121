"""
app.py — Streamlit YOLO video processor with crosshair + CSV + live chart

Requirements:
    pip install streamlit opencv-python-headless ultralytics pandas

Your custom YOLO model path is fixed:
    C:/Users/SRG Project/venv1/Project1/SFM/runsv1/detect/train11/weights/best.pt
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = r"C:\Users\SRG Project\venv1\Project1\SFM\runsv1\detect\train11\weights\best.pt"

st.set_page_config(page_title="YOLO Video Processor", layout="wide")

# ============================================================
# HELPERS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    """Load your custom YOLO model (cached)."""
    return YOLO(MODEL_PATH)

def draw_crosshair(img, center, size=10, color=(0, 255, 0), thickness=1):
    """Draw a crosshair at given (x, y)."""
    x, y = int(center[0]), int(center[1])
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)

def draw_bbox_and_crosshair(img, xyxy, label=None, conf=None):
    """Draw bbox + crosshair, return center coords."""
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    draw_crosshair(img, (cx, cy))
    if label is not None:
        text = f"{label} {conf:.2f}" if conf is not None else label
        cv2.putText(img, text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return (cx, cy)

# ============================================================
# UI
# ============================================================
st.title("Vibration Analysis📈")

# st.markdown(f"**Model path:** `{MODEL_PATH}`")

conf_thres = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
max_frames = st.number_input("Max frames to process (0 = all)", min_value=0, value=0, step=1)
show_fps = st.checkbox("Show FPS on video", value=True)

# st.markdown("### Capture / Upload Video")
cam_file = None
# cam_file = st.camera_input("Record video (mobile) or take short video", key="cam_input")
# st.write("OR")
upload_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv", "webm"])

process_btn = st.button("Start Processing")

video_display = st.empty()
chart_display = st.empty()
progress_bar = st.progress(0)
status_text = st.empty()

# ============================================================
# PROCESS
# ============================================================
if process_btn:
    # Choose video source
    video_bytes = None
    filename = None

    if cam_file is not None:
        video_bytes = cam_file.read()
        filename = getattr(cam_file, "name", "camera_video.mp4")
    elif upload_file is not None:
        video_bytes = upload_file.read()
        filename = getattr(upload_file, "name", "uploaded_video.mp4")
    else:
        st.warning("Please record or upload a video first.")
        st.stop()

    tmp_video = Path(tempfile.gettempdir()) / f"input_{int(time.time())}_{filename}"
    tmp_video.write_bytes(video_bytes)

    # Load model
    status_text.info("Loading custom YOLO model...")
    model = load_yolo_model()
    model.conf = conf_thres
    status_text.success("Model loaded successfully.")

    # Video capture
    cap = cv2.VideoCapture(str(tmp_video))
    if not cap.isOpened():
        st.error("Could not open video file.")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = total_frames if max_frames == 0 else min(max_frames, total_frames)

    out_video = Path(tempfile.gettempdir()) / f"processed_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

    # Prepare storage
    data_rows = []
    det_counts = []
    last_time = time.time()

    # Initialize chart
    # chart_df = pd.DataFrame({"frame": [], "detections": []})
    # chart_plot = chart_display.line_chart(chart_df.set_index("frame"))
    # Initialize chart for X & Y center displacement
    chart_df = pd.DataFrame({"frame": [], "x_center": [], "y_center": []})
    chart_plot = chart_display.line_chart(chart_df.set_index("frame"))


    status_text.info(f"Processing {limit} frames...")

    for frame_idx in range(limit):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO
        results = model(frame, imgsz=640, verbose=False)
        r = results[0]

        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []

        for i, box in enumerate(boxes):
            cls_id = int(cls_ids[i]) if len(cls_ids) > i else None
            conf = float(confs[i]) if len(confs) > i else 0.0
            cls_name = model.names.get(cls_id, str(cls_id)) if cls_id is not None else ""
            cx, cy = draw_bbox_and_crosshair(frame, box, cls_name, conf)
            data_rows.append({
                "frame": frame_idx,
                "class_id": cls_id,
                "class_name": cls_name,
                "x_center": cx,
                "y_center": cy,
                "confidence": conf
            })

        det_counts.append(len(boxes))

        # FPS overlay
        if show_fps:
            now = time.time()
            fps_val = 1.0 / (now - last_time) if now - last_time > 0 else 0.0
            last_time = now
            cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Write and preview
        writer.write(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_display.image(rgb, channels="RGB", use_container_width =True)

        # Update chart
        # chart_df = pd.DataFrame({"frame": [frame_idx], "detections": [len(boxes)]})
        # chart_plot.add_rows(chart_df.set_index("frame"))
        # Update chart with X and Y center displacement
        if len(boxes) > 0:
            # take first detected box (or average if multiple)
            cx_vals = [draw_bbox_and_crosshair(frame, box)[0] for box in boxes]
            cy_vals = [draw_bbox_and_crosshair(frame, box)[1] for box in boxes]
            cx_mean, cy_mean = np.mean(cx_vals), np.mean(cy_vals)
            chart_df = pd.DataFrame({"frame": [frame_idx], "x_center": [cx_mean], "y_center": [cy_mean]})
            chart_plot.add_rows(chart_df.set_index("frame"))



        # Update progress
        if limit > 0:
            progress_bar.progress(int((frame_idx + 1) / limit * 100))
        status_text.info(f"Frame {frame_idx + 1}/{limit} — {len(boxes)} detections")

        time.sleep(0.01)

    cap.release()
    writer.release()

    status_text.success("Processing complete!")

    # Save CSV
    df = pd.DataFrame(data_rows)
    csv_file = Path(tempfile.gettempdir()) / f"detections_{int(time.time())}.csv"
    df.to_csv(csv_file, index=False)

    # Show results
    st.markdown("### Processed Video")
    st.video(str(out_video))

    st.markdown("### Detection CSV Preview")
    st.dataframe(df.head(50))

    with open(csv_file, "rb") as f:
        st.download_button("⬇️ Download CSV", data=f, file_name="detections.csv", mime="text/csv")

    st.markdown("### Detections per Frame")
    det_df = pd.DataFrame({"frame": list(range(len(det_counts))), "detections": det_counts})
    st.line_chart(det_df.set_index("frame"))

    st.info(f"Video saved to: {out_video}")
    st.info(f"CSV saved to: {csv_file}")

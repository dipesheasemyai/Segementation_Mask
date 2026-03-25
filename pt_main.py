"""
YOLO PT Glove Cropping Pipeline
-------------------------------

Purpose:
    Detect gloves in video using a YOLO .pt model, crop detected regions,
    save them, and display live annotated preview.

Features:
    - Hard-coded video, model, and output paths
    - Modular functions: model loading, video capture, processing
    - Confidence threshold control
    - Auto folder creation
"""

import os
import cv2
from ultralytics import YOLO


# ----------------------------
# Step 1: Load YOLO Model and Video
# ----------------------------
def load_model(model_path):
    """
    Load YOLO .pt model.
    """
    return YOLO(model_path)

def load_video(video_path):
    """
    Load video capture object.
    """
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise RuntimeError("Error: Could not open video file.")
    return vid

# ----------------------------
# Step 2: Process video frames
# ----------------------------
def process_video(video_path, model_path):
    """
    Read video frames, detect gloves, crop and save high-confidence boxes.
    """
    OUTPUT_FOLDER = "cropped_image"
    CONF_THRESHOLD = 0.9  # Only save boxes above this confidence
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    vid = load_video(video_path)
    model = load_model(model_path)
    image_count = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        resized_frame = cv2.resize(annotated_frame, (800, 600))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf < CONF_THRESHOLD:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                filename = os.path.join(OUTPUT_FOLDER, f"crop_{image_count}.jpg")
                cv2.imwrite(filename, crop)
                image_count += 1

        # Display live annotated preview
        cv2.imshow("Video Preview", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

# ----------------------------
# Step 3: Run prcess_video
# ----------------------------
if __name__ == "__main__":
    VIDEO_PATH = '/home/easemyai/Documents/object_seg/DPL_Sample_Video/Cam 192_168_1_17/20260317105131.ts'
    MODEL_PATH = '/home/easemyai/Documents/object_seg/model/best.pt'
    process_video(VIDEO_PATH, MODEL_PATH)
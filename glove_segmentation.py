"""
YOLO Glove Segmentation Pipeline (Hard-coded Paths)
---------------------------------------------------

Purpose:
    Detect gloves in a video using a YOLO model, apply brown tint,
    place on white background, and save cropped images.

Features:
    - Hard-coded video and model paths
    - Modular functions for mask processing and video loop
    - Live preview of detections

Requirements:
    - numpy
    - opencv-python
    - ultralytics
    - os
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Step 1: Glove mask processing function
# ----------------------------
def process_glove(crop, mask):
    """
    Apply brown tint to glove crop and place it on a white background.

    Parameters:
        crop (np.ndarray): Cropped glove image (BGR)
        mask (np.ndarray): Binary mask corresponding to glove (0-1 or 0-255)

    Returns:
        np.ndarray: Processed glove image
    """
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]))

    # Convert crop to LAB color space for brown tint
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    new_a = np.full_like(a, 145)
    new_b = np.full_like(b, 155)
    brown_lab = cv2.merge([l, new_a, new_b])
    brown_bgr = cv2.cvtColor(brown_lab, cv2.COLOR_LAB2BGR)

    # White background
    white_bg = np.full(crop.shape, 255, dtype=np.uint8)

    # Smooth mask for alpha blending
    mask_factor = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
    mask_factor = np.stack([mask_factor] * 3, axis=-1)

    final = (brown_bgr * mask_factor) + (white_bg * (1.0 - mask_factor))
    return np.clip(final, 0, 255).astype(np.uint8)

# ----------------------------
# Step 2: Video processing function
# ----------------------------
def process_video(video_path, model_path):
    """
    Process video frames using YOLO, apply mask processing, save cropped gloves.
    """
    CONF_THRESHOLD = 0.7
    OUTPUT_FOLDER = 'cropped_image'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    model = YOLO(model_path)
    vid = cv2.VideoCapture(video_path)
    image_count = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            if r.masks is not None:
                for i, mask_data in enumerate(r.masks.data):
                    # Get bounding box and confidence
                    x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
                    conf = float(r.boxes.conf[i])
                    if conf < CONF_THRESHOLD:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    mask_full = mask_data.cpu().numpy()
                    mask_resized = cv2.resize(mask_full, (frame.shape[1], frame.shape[0]))
                    mask_crop = mask_resized[y1:y2, x1:x2]

                    if np.any(mask_crop > 0):
                        processed_img = process_glove(crop, mask_crop)
                        filename = os.path.join(OUTPUT_FOLDER, f"crop_{image_count}.jpg")
                        cv2.imwrite(filename, processed_img)
                        image_count += 1

        cv2.imshow("Preview", results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

# ----------------------------
# Step 3: Run process_video
# ----------------------------
if __name__ == "__main__":
    VIDEO_PATH = '/home/easemyai/Documents/object_seg/DPL_Sample_Video/Cam 192_168_1_17/20260317105131.ts'
    MODEL_PATH = '/home/easemyai/Documents/object_seg/model/best.pt'
    process_video(VIDEO_PATH, MODEL_PATH)
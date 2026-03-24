import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load YOLO model with bbox + segmentation
model = YOLO("/home/easemyai/Documents/object_seg/model/best.pt")

# Output folder
output_folder = "brown_gloves"
os.makedirs(output_folder, exist_ok=True)

# Video path
cap = cv2.VideoCapture('/home/easemyai/Documents/object_seg/DPL_Sample_Video/Cam 192_168_1_17/20260317105131.ts')

img_count = 0

# Define brown color in BGR (OpenCV uses BGR)
brown_color = (42, 42, 165)  # Dark brown in BGR

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        boxes = r.boxes
        masks = r.masks

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf <= 0.8:  # high-confidence filter
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Get segmentation mask as numpy array
            mask = masks.data[i]           # PyTorch tensor
            mask_np = mask.cpu().numpy()   # convert to NumPy
            mask_resized = cv2.resize(mask_np, (crop.shape[1], crop.shape[0]))
            mask_resized = (mask_resized * 255).astype(np.uint8)

            # Create white background
            white_bg = np.ones_like(crop, dtype=np.uint8) * 255

            # Create brown image same size as crop
            brown_fg = np.ones_like(crop, dtype=np.uint8)
            brown_fg[:] = brown_color

            # Apply mask: brown on glove, white elsewhere
            fg = cv2.bitwise_and(brown_fg, brown_fg, mask=mask_resized)
            inv_mask = cv2.bitwise_not(mask_resized)
            bg = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
            final = cv2.add(fg, bg)

            # Save the result
            filename = os.path.join(output_folder, f"glove_{img_count}.jpg")
            cv2.imwrite(filename, final)
            img_count += 1

    # Optional: show video with detection boxes
    annotated = results[0].plot()
    resized = cv2.resize(annotated, (800, 600))
    cv2.imshow("Brown Glove Detection", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total brown glove images saved: {img_count}")
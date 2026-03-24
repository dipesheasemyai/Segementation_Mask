import cv2
import numpy as np
import onnxruntime as ort
import os


def onnx_setup(onnx_path):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    model_inputs = session.get_inputs()
    in_h, in_w = model_inputs[0].shape[2], model_inputs[0].shape[3]

    return in_h, in_w, model_inputs, session
    

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resizes image to a square with gray padding to maintain aspect ratio."""
    shape = im.shape[:2] # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im, r, (dw, dh)


def process_frame(frame, in_w, in_h, model_inputs, session):
    CONF_THRESH = 0.65
    IOU_THRESH = 0.45
    orig_h, orig_w = frame.shape[:2]

    # 1. Letterbox (Adds gray bars to keep glove shape perfect)
    input_img, ratio, (pad_w, pad_h) = letterbox(frame, (in_w, in_h))
    blob = input_img.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]

    # Inference
    outputs = session.run(None, {model_inputs[0].name: blob})
    preds = np.squeeze(outputs[0]).T 
    protos = np.squeeze(outputs[1]) 

    boxes = preds[:, :4]
    scores = np.max(preds[:, 4:-32], axis=1)
    mask_coeffs = preds[:, -32:]

    idx = np.where(scores > CONF_THRESH)[0]
    nms_boxes, nms_scores, nms_coeffs = [], [], []
    for i in idx:
        xc, yc, w, h = boxes[i]
        # Convert from letterbox square back to original pixel coordinates
        x1 = (xc - w/2 - pad_w) / ratio
        y1 = (yc - h/2 - pad_h) / ratio
        w_scaled = w / ratio
        h_scaled = h / ratio
        nms_boxes.append([float(x1), float(y1), float(w_scaled), float(h_scaled)])
        nms_scores.append(float(scores[i]))
        nms_coeffs.append(mask_coeffs[i])

    indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, CONF_THRESH, IOU_THRESH)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = nms_boxes[i]
            x1, y1 = int(max(0, x)), int(max(0, y))
            x2, y2 = int(min(orig_w, x + w)), int(min(orig_h, y + h))

            crop_img = frame[y1:y2, x1:x2].copy()
            if crop_img.size == 0: continue

            # --- MASK GENERATION ---
            coeffs = nms_coeffs[i]
            c, mh, mw = protos.shape
            mask_raw = (coeffs @ protos.reshape(c, -1)).reshape(mh, mw)
            mask_raw = sigmoid(mask_raw)

            # Resize mask back to 640x640, then remove padding, then scale to original
            mask_full = cv2.resize(mask_raw, (in_w, in_h))
            mask_full = mask_full[int(pad_h):int(in_h-pad_h), int(pad_w):int(in_w-pad_w)]
            mask_full = cv2.resize(mask_full, (orig_w, orig_h))
            mask_crop = mask_full[y1:y2, x1:x2]

            # --- SMOOTHING & PURE WHITE BACKGROUND ---
            _, mask_binary = cv2.threshold(mask_crop, 0.5, 1.0, cv2.THRESH_BINARY)
            mask_feathered = cv2.GaussianBlur(mask_binary, (9, 9), 0)
            
            alpha = np.stack([mask_feathered] * 3, axis=-1).astype(np.float32)
            white_bg = np.full(crop_img.shape, 255, dtype=np.uint8)
            
            final_res = (crop_img.astype(np.float32) * alpha) + (white_bg.astype(np.float32) * (1.0 - alpha))
            final_res = np.clip(final_res, 0, 255).astype(np.uint8)

            results.append({
                'clean_crop': final_res,
                'box': (x1, y1, x2, y2),
                'conf': nms_scores[i]
            })
            
    return results

def vid_processing(vid_path, onnx_path):
    output_dir = "perfect_glove_crops"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup ONNX
    session, model_inputs, in_h, in_w = onnx_setup(onnx_path)
    
    # Setup Video
    vid = cv2.VideoCapture(vid_path)
    image_count = 0

    print("Processing video... Press 'q' to stop.")

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Process the current frame
        detections = process_frame(frame, session, model_inputs, in_h, in_w)

        for det in detections:
            # 1. Save the processed glove image
            save_path = os.path.join(output_dir, f"glove_{image_count}.jpg")
            cv2.imwrite(save_path, det['clean_crop'])
            image_count += 1
            
            # 2. Draw visual feedback on the preview frame
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Glove: {det['conf']:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show the video feed
        cv2.imshow("Detection Preview", cv2.resize(frame, (1280, 720)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vid_path = '/home/easemyai/Documents/object_seg/DPL_Sample_Video/Cam 192_168_1_17/20260317105131.ts'
    onnx_path = "/home/easemyai/Documents/object_seg/model/best.onnx"
    vid_processing(vid_path, onnx_path)
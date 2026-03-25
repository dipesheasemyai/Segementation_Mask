"""
Video Inference using TensorRT Engine
-------------------------------------

Purpose:
    This script runs video frames through a TensorRT-optimized deep learning model 
    for fast object segmentation (e.g., detecting gloves). Supports GPU acceleration 
    and FP16 precision.

Requirements:
    - cuda
    - os
    - opencv-python (cv2)
    - numpy
    - tensorrt (TensorRT Python API)

Usage:
    Modify the video path and engine path at the bottom, then run:
        python engine_main.py
    
"""


import cv2
import os
import numpy as np
import tensorrt as trt
from cuda import cudart as cuda

# ----------------------------
# Step 1: TensorRT Engine Class
# ----------------------------

class TRTEngine:
    """
    Load a TensorRT engine and provides a callable interface for inference.
    
    Attributes:
        logger: TensorRT logger instance.
        runtime: TensorRT runtime object.
        engine: Deserialized TensorRT engine.
        context: Execution context for inference.
        inputs: List of input tensor info and device allocation.
        outputs: List of output tensor info and device allocation.
        bindings: List of device pointers for engine bindings. 
    """
    def __init__(self, engine_path):
        """
        Load a TensorRT engine from file and allocate GPU memory for inputs and outputs.

        Parameters:
            engine_path (str): Path to the TensorRT .engine file
        """
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = list(self.engine.get_tensor_shape(name))
            if shape[0] == -1: shape[0] = 1 
            
            size = trt.volume(shape) * np.dtype(trt.nptype(dtype)).itemsize
            _, device_mem = cuda.cudaMalloc(size)
            
            binding = {'name': name, 'dtype': trt.nptype(dtype), 'shape': tuple(shape), 
                      'allocation': device_mem, 'size': size}
            self.bindings.append(device_mem)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def __call__(self, blob):
        """
        Run inference on the input blob.

        Parameters:
            blob (np.ndarray): Preprocessed input tensor (NCHW, float32)

        Returns:
            list: List of output arrays corresponding to engine outputs
        """
        # Copy input to GPU
        cuda.cudaMemcpy(self.inputs[0]['allocation'], blob.ctypes.data, 
                        self.inputs[0]['size'], cuda.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
        # Bind tensor to context
        for b in self.inputs + self.outputs:
            self.context.set_tensor_address(b['name'], b['allocation'])
        
        # Execute inference aysnchronously
        self.context.execute_async_v3(0)
        
        # Copy outputs back to gpu
        results = []
        for out in self.outputs:
            out_data = np.zeros(out['shape'], dtype=out['dtype'])
            cuda.cudaMemcpy(out_data.ctypes.data, out['allocation'], 
                            out['size'], cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            results.append(out_data)
        return results

# ----------------------------
# Step 2: Sigmoid Function
# ----------------------------

def sigmoid(x): 
    """
    Apply sigmoid activation function.

    Parameters:
        x (np.ndarray): Input array

    Returns:
        np.ndarray: Sigmoid of input
    """
    return 1 / (1 + np.exp(-x))


# ----------------------------
# Step 3: Preprocess video frame
# ----------------------------

def process_frame(frame, in_w, in_h, engine):
    """
    Preprocess a video frame, run inference, and extract masks and boxes.

    Parameters:
        frame (np.ndarray): Original BGR video frame
        in_w (int): Model input width
        in_h (int): Model input height
        engine (TRTEngine): TensorRT engine object

    Returns:
        list: List of detection results, each as a dict with keys:
            - 'mask': Binary mask
            - 'box': Bounding box coordinates (x1, y1, x2, y2)
            - 'conf': Confidence score
            - 'crop': Cropped image of detected object
    """

    # BALANCE POINT: 0.65 catches gloves without blinking too much
    CONF_THRESH = 0.8  
    IOU_THRESH = 0.45
    
    orig_h, orig_w = frame.shape[:2]
    r = min(in_w/orig_w, in_h/orig_h)
    nw, nh = int(orig_w*r), int(orig_h*r)
    
    # Resize and pad frame to model input size
    im_resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
    pad_w, pad_h = (in_w - nw) // 2, (in_h - nh) // 2
    canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = im_resized

    # Convert to NCHW and normalize
    blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)

    # Run inference
    outputs = engine(blob)
    preds = np.squeeze(outputs[0])
    if preds.shape[0] < preds.shape[1]: preds = preds.T 
    protos = np.squeeze(outputs[1])

    boxes = preds[:, :4]
    scores = np.max(preds[:, 4:-32], axis=1)
    mask_coeffs = preds[:, -32:]

    # Apply Non-Max Supperssion
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESH, IOU_THRESH)

    results = []
    if len(indices) > 0:
        # Pre-calculate Proto reshaping to save CPU time
        c, mh, mw = protos.shape
        protos_flat = protos.reshape(c, -1)
        
        for i in indices.flatten():
            xc, yc, w, h = boxes[i]
            x1 = int((xc - w/2 - pad_w) / r)
            y1 = int((yc - h/2 - pad_h) / r)
            x2, y2 = int(x1 + (w/r)), int(y1 + (h/r))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)

            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size < 100: continue

            # Compute MASK: Only compute for the detected box
            m_coeffs = mask_coeffs[i]
            mask_raw = sigmoid(m_coeffs @ protos_flat).reshape(mh, mw)
            
            # Efficient resizing: Resize directly to the crop size
            mask_resized = cv2.resize(mask_raw, (in_w, in_h))
            mask_final = mask_resized[pad_h:pad_h+nh, pad_w:pad_w+nw]
            mask_final = cv2.resize(mask_final, (orig_w, orig_h))
            mask_crop = mask_final[y1:y2, x1:x2]

            _, mask_bin = cv2.threshold(mask_crop, 0.5, 255, cv2.THRESH_BINARY)
            
            results.append({
                'mask': mask_bin.astype(np.uint8),
                'box': (x1, y1, x2, y2),
                'conf': scores[i],
                'crop': crop_img
            })
    return results


# ----------------------------
# Step 4: Video processing loop
# ----------------------------

def vid_processing(vid_path, engine_path):
    """
    Process a video file frame-by-frame using a TensorRT engine, saving
    high-confidence glove crops and visualizing results.

    Parameters:
        vid_path (str): Path to the input video
        engine_path (str): Path to the TensorRT engine file
    """

    output_dir = "perfect_glove_crops"
    os.makedirs(output_dir, exist_ok=True)
    engine = TRTEngine(engine_path)
    in_h, in_w = engine.inputs[0]['shape'][2], engine.inputs[0]['shape'][3]
    
    vid = cv2.VideoCapture(vid_path)
    # Skip frames to speed up preview if needed (e.g., process every 2nd frame)
    frame_count = 0

    while True:
        ret, frame = vid.read()
        if not ret: break

        detections = process_frame(frame, in_w, in_h, engine)

        for det in detections:
            x1, y1, x2, y2 = det['box']
            # Only save if confidence is decent
            if det['conf'] > 0.90:
                # White background crop logic
                mask_3d = cv2.merge([det['mask']]*3) / 255.0
                fg = (det['crop'] * mask_3d).astype(np.uint8)
                bg = (np.full(det['crop'].shape, 255) * (1 - mask_3d)).astype(np.uint8)
                final_res = cv2.add(fg, bg)
                
                cv2.imwrite(os.path.join(output_dir, f"g_{frame_count}.jpg"), final_res)
                frame_count += 1
            
            # Draw bounding box and confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Glove: {det['conf']:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show smaller window for speed
        cv2.imshow("TRT Fast Preview", cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vid.release()
    cv2.destroyAllWindows()


# ----------------------------
# Step 5: Run Video Processing
# ----------------------------

if __name__ == "__main__":
    v_path = '/home/easemyai/Documents/object_seg/DPL_Sample_Video/Cam 192_168_1_17/20260317105131.ts'
    e_path = "/home/easemyai/Documents/object_seg/model/best.engine"
    vid_processing(v_path, e_path)
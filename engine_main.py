import cv2
import os
import numpy as np
import tensorrt as trt
from cuda import cudart as cuda

class TRTEngine:
    def __init__(self, engine_path):
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
        # Ensure blob is float32 and contiguous for the GPU
        blob = np.ascontiguousarray(blob.astype(np.float32))
        cuda.cudaMemcpy(self.inputs[0]['allocation'], blob.ctypes.data, 
                        self.inputs[0]['size'], cuda.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
        for b in self.inputs + self.outputs:
            self.context.set_tensor_address(b['name'], b['allocation'])
        self.context.execute_async_v3(0)
        
        results = []
        for out in self.outputs:
            out_data = np.zeros(out['shape'], dtype=out['dtype'])
            cuda.cudaMemcpy(out_data.ctypes.data, out['allocation'], 
                            out['size'], cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            results.append(out_data)
        return results

def sigmoid(x): return 1 / (1 + np.exp(-x))

def process_frame(frame, in_w, in_h, engine):
    # --- STABILITY SETTINGS ---
    CONF_THRESH = 0.55  # Lowered to prevent "missing" gloves
    IOU_THRESH = 0.45
    MIN_AREA = 2500     # Ignore tiny "flicker" detections
    
    orig_h, orig_w = frame.shape[:2]
    r = min(in_w/orig_w, in_h/orig_h)
    nw, nh = int(orig_w*r), int(orig_h*r)
    
    # Preprocessing
    im_resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
    pad_w, pad_h = (in_w - nw) // 2, (in_h - nh) // 2
    canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = im_resized
    blob = canvas.transpose(2, 0, 1)[None] / 255.0

    outputs = engine(blob)
    preds = np.squeeze(outputs[0])
    if preds.shape[0] < preds.shape[1]: preds = preds.T 
    protos = np.squeeze(outputs[1])

    boxes = preds[:, :4]
    # Handle multi-class or single class scores safely
    scores = np.max(preds[:, 4:-32], axis=1) if preds.shape[1] > 36 else preds[:, 4]
    mask_coeffs = preds[:, -32:]

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESH, IOU_THRESH)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            # Coordinate scaling
            xc, yc, w, h = boxes[i]
            
            # STABILITY CHECK: Size Filter
            if (w/r * h/r) < MIN_AREA: continue

            x1 = int((xc - w/2 - pad_w) / r)
            y1 = int((yc - h/2 - pad_h) / r)
            x2, y2 = int(x1 + (w/r)), int(y1 + (h/r))
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)

            crop_img = frame[y1:y2, x1:x2].copy()
            if crop_img.size < 10: continue

            # --- MASK GENERATION ---
            c, mh, mw = protos.shape
            mask_raw = sigmoid((mask_coeffs[i] @ protos.reshape(c, -1)).reshape(mh, mw))
            mask_full = cv2.resize(mask_raw, (in_w, in_h))
            mask_full = mask_full[pad_h:pad_h+nh, pad_w:pad_w+nw]
            mask_full = cv2.resize(mask_full, (orig_w, orig_h))
            mask_crop = mask_full[y1:y2, x1:x2]

            _, mask_binary = cv2.threshold(mask_crop, 0.5, 1.0, cv2.THRESH_BINARY)
            alpha = np.stack([cv2.GaussianBlur(mask_binary, (7, 7), 0)] * 3, axis=-1).astype(np.float32)
            white_bg = np.full(crop_img.shape, 255, dtype=np.uint8)
            final_res = (crop_img.astype(np.float32) * alpha) + (white_bg.astype(np.float32) * (1.0 - alpha))
            
            results.append({
                'clean_crop': np.clip(final_res, 0, 255).astype(np.uint8),
                'box': (x1, y1, x2, y2),
                'conf': scores[i]
            })
    return results

def vid_processing(vid_path, engine_path):
    output_dir = "perfect_glove_crops"
    os.makedirs(output_dir, exist_ok=True)
    engine = TRTEngine(engine_path)
    in_h, in_w = engine.inputs[0]['shape'][2], engine.inputs[0]['shape'][3]
    
    vid = cv2.VideoCapture(vid_path)
    image_count = 0

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret: break

        detections = process_frame(frame, in_w, in_h, engine)

        for det in detections:
            # 1. Save Crop
            cv2.imwrite(os.path.join(output_dir, f"glove_{image_count}.jpg"), det['clean_crop'])
            image_count += 1
            
            # 2. Draw Stable UI
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{image_count} CF:{det['conf']:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Stable TensorRT Inference", cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    v_path = '/home/easemyai/Documents/object_seg/DPL_Sample_Video/Cam 192_168_1_17/20260317105131.ts'
    e_path = "/home/easemyai/Documents/object_seg/model/best.engine"
    vid_processing(v_path, e_path)
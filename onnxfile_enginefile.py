"""
ONNX to TensorRT Engine Conversion
----------------------------------

Purpose:
    Convert a trained ONNX model (.onnx) into a TensorRT engine (.engine)
    for optimized GPU inference. This improves performance for deployment
    on NVIDIA GPUs, supporting FP32/FP16 and memory optimization.

Requirements:
    - tensorrt (TensorRT Python API)
    - A valid ONNX model file

Outputs:
    - Serialized TensorRT engine file (.engine) ready for inference
"""

import tensorrt as trt
import os

# ----------------------------
# Step 1: Set paths & logger
# ----------------------------
ONNX_PATH = "/home/easemyai/Documents/object_seg/model/best.onnx"
ENGINE_PATH = "/home/easemyai/Documents/object_seg/model/best.engine"

# TensorRT logger to capture warnings/errors during engine build
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ----------------------------
# Step 2: Load ONNX model
# ----------------------------
# Read the ONNX model file as binary
with open(ONNX_PATH, "rb") as f:
    onnx_model = f.read()

# ----------------------------
# Step 3: Create TensorRT builder, network, and parser
# ----------------------------
# Builder: used to construct the TensorRT engine
builder = trt.Builder(TRT_LOGGER)

# Network: defines the computation graph, Explicit batch mode is required for modern ONNX models
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Parser: converts the ONNX model into TensorRT network representation
parser = trt.OnnxParser(network, TRT_LOGGER)

# ----------------------------
# Step 4: Parse ONNX model into TensorRT network
# ----------------------------
if not parser.parse(onnx_model):
    print("ERROR: Failed to parse the ONNX file.")
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    exit(1)

# ----------------------------
# Step 5: Configure builder settings
# ----------------------------
config = builder.create_builder_config()

# Set maximum GPU workspace memory (1 GB here) for optimization
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# Optional: You could enable FP16 or INT8 here for faster inference:
# config.set_flag(trt.BuilderFlag.FP16)

# ----------------------------
# Step 6: Build serialized engine
# ----------------------------
print("Building and serializing engine... this may take a few minutes.")

# Build serialized network (engine plan) directly; no separate build_engine() call needed
plan = builder.build_serialized_network(network, config)

if plan is None:
    print("ERROR: Could not build the engine plan.")
    exit(1)

# ----------------------------
# Step 7: Save engine to file
# ----------------------------
with open(ENGINE_PATH, "wb") as f:
    f.write(plan)

print(f"Success! TensorRT engine saved to {ENGINE_PATH}")
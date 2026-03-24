import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ONNX_PATH = "/home/easemyai/Documents/object_seg/model/best.onnx"
ENGINE_PATH = "/home/easemyai/Documents/object_seg/model/best.engine"

with open(ONNX_PATH, "rb") as f:
    onnx_model = f.read()

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse the model and check for errors
if not parser.parse(onnx_model):
    print("ERROR: Failed to parse the ONNX file.")
    for error in range(parser.num_errors):
        print(parser.get_error(error))
    exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB

# --- THE TENSORRT 10 WAY ---
print("Building and serializing engine... this will take a few minutes.")

# This one line replaces build_engine() and engine.serialize()
plan = builder.build_serialized_network(network, config)

if plan is None:
    print("ERROR: Could not build the engine plan.")
    exit(1)

with open(ENGINE_PATH, "wb") as f:
    f.write(plan)

print(f"✅ Success! Engine saved to {ENGINE_PATH}")
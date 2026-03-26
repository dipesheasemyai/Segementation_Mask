import argparse
import os
import sys
from utils import pt_inference, onnx_inference, engine_inference

def main():
    parser = argparse.ArgumentParser(description="Model testing tools")
    parser.add_argument("--format", required=True, choices=['pt', 'onnx', 'engine'], help="Model Format")
    parser.add_argument("--path", type=str, help = "file path (requried)")
    parser.add_argument("--vid_path",type=str, help = "video file path (requried)")
    args = parser.parse_args()

    try:
        
        if not os.path.exists(args.vid_path):
              raise FileNotFoundError("Video path is not exist.")

        if not os.path.exists(args.path):
              raise FileNotFoundError("Model path is not exist.")

        if  args.format == "pt":
                print("--Running Pytorch file--")
                pt_inference.process_video(args.vid_path, args.path)

        elif  args.format == "onnx":
                print("--Running onnx file--")    
                onnx_inference.vid_processing(args.vid_path, args.path)

        elif args.format == "engine":
                print("--Running TensorRT file--")
                engine_inference.vid_processing(args.vid_path, args.path)
        
    
    except FileNotFoundError as e:
        print(f"PATH ERROR: {e}")
        sys.exit(1)
    
    


if __name__ == "__main__":
    main()

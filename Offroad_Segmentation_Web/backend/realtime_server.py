from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import torch
import numpy as np
import base64
import json
import asyncio
import onnxruntime as ort
from model_utils import smp, mask_to_rgb
from astar_planner import AStarPlanner

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ONNX_PATH = 'best_model.onnx'

# Global model and planners
ort_session = None
planner = AStarPlanner()

@app.on_event("startup")
async def startup_event():
    global ort_session
    print(f"Loading ONNX model from {ONNX_PATH}...")
    try:
        # Prefer CUDA if available, else CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(ONNX_PATH, providers=providers)
        print(f"ONNX Model loaded successfully with providers: {ort_session.get_providers()}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")

@app.websocket("/ws/pathfinder")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to Real-Time PathFinder")
    # Preprocessing fn from SMP (ResNet50)
    preprocess = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')
    
    try:
        while True:
            # Receive base64 frame from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if "frame" not in message:
                continue
                
            frame_b64 = message["frame"]
            # Decode base64 to bytes
            frame_bytes = base64.b64decode(frame_b64.split(",")[-1])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
                
            # Preprocess image to 320x320
            resized = cv2.resize(image, (320, 320))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            preprocessed = preprocess(rgb).astype(np.float32)
            input_tensor = preprocessed.transpose(2, 0, 1).reshape(1, 3, 320, 320)
            
            # 1. Run ONNX Inference
            outputs = ort_session.run(None, {'input': input_tensor})
            mask_logits = outputs[0][0]
            mask_indices = np.argmax(mask_logits, axis=0).astype(np.uint8)
            
            # 2. Colorize mask for visualization (320x320)
            mask_rgb = mask_to_rgb(mask_indices)
            mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
            # Resize mask to 160x160 to save bandwidth if needed, but let's try 320 first
            _, mask_buffer = cv2.imencode('.jpg', mask_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
            mask_b64 = base64.b64encode(mask_buffer).decode('utf-8')
            
            # 3. Downsample mask for A* speed (320x320 -> 160x160)
            DS_FACTOR = 2
            mask_ds = cv2.resize(mask_indices, (320 // DS_FACTOR, 320 // DS_FACTOR), interpolation=cv2.INTER_NEAREST)
            
            # 4. Run A* Pathfinding on downsampled mask
            path_ds = planner.find_path(mask_ds)
            
            # 5. Scale path back to 320x320 for frontend
            path = [(int(x * DS_FACTOR), int(y * DS_FACTOR)) for x, y in path_ds]
            
            response = {
                "path": path,
                "mask": mask_b64,
                "maskWidth": 320,
                "maskHeight": 320
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

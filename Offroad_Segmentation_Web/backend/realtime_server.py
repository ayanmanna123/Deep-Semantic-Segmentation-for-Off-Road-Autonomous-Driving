from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import torch
import numpy as np
import base64
import json
import asyncio
from model_utils import load_model, predict_on_image
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
MODEL_PATH = '../../Offroad_Segmentation_Scripts/runs/checkpoints/best_model.pth'

# Global model and planners
model = None
preprocessing_fn = None
planner = AStarPlanner()

@app.on_event("startup")
async def startup_event():
    global model, preprocessing_fn
    print(f"Loading model from {MODEL_PATH} for Real-Time Server...")
    try:
        model, preprocessing_fn = load_model(MODEL_PATH, DEVICE)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.websocket("/ws/pathfinder")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to Real-Time PathFinder")
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
            
            # 1. Run inference
            orig_img, mask_rgb, mask_indices = predict_on_image(model, preprocessing_fn, frame_bytes, DEVICE)
            
            # 2. Run A* Pathfinding
            # We use a downsampled mask for speed if needed, but 320x320 should be okay
            path = planner.find_path(mask_indices)
            
            # 3. Prepare response
            # Scaling path back to original image size if needed (but frontend can scale too)
            # For now, let's send the path as a list of coordinates
            
            response = {
                "path": path,
                "maskWidth": mask_indices.shape[1],
                "maskHeight": mask_indices.shape[0]
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

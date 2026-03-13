from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
import cv2
import torch
import numpy as np
import base64
import os
from model_utils import load_model, predict_on_image, get_overlay, get_class_stats, process_gt_mask, mask_to_rgb, get_path_visualization

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '../../Offroad_Segmentation_Scripts/runs/checkpoints/best_model.pth'
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Offroad_Segmentation_Scripts/runs/plots'))

# Mount static directory for plots
if os.path.exists(PLOTS_DIR):
    app.mount("/static/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

# Global model and preprocessing function
model = None
preprocessing_fn = None

@app.on_event("startup")
async def startup_event():
    global model, preprocessing_fn
    print(f"Loading model from {MODEL_PATH}...")
    try:
        if os.path.exists(MODEL_PATH):
            model, preprocessing_fn = load_model(MODEL_PATH, DEVICE)
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model path {MODEL_PATH} does not exist. Inference will fail.")
    except Exception as e:
        print(f"Error loading model: {e}")

def to_b64(img):
    # Convert RGB to BGR for cv2 encoding
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Run prediction
        orig_img, mask_rgb, mask_indices = predict_on_image(model, preprocessing_fn, contents, DEVICE)
        
        # Generate path visualization
        path_viz = get_path_visualization(orig_img, mask_indices, mask_rgb)
        
        return {
            "original": to_b64(orig_img),
            "mask": to_b64(mask_rgb),
            "path_viz": to_b64(path_viz)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze")
async def analyze(colour_image: UploadFile = File(...), segment_image: UploadFile = File(...)):
    print(f"Received Analyze Request:")
    print(f" - colour_image: {colour_image.filename if colour_image else 'None'}")
    print(f" - segment_image: {segment_image.filename if segment_image else 'None'}")
    try:
        img_contents = await colour_image.read()
        gt_contents = await segment_image.read()
        
        # Run prediction
        orig_img, pred_mask_rgb, pred_mask = predict_on_image(model, preprocessing_fn, img_contents, DEVICE)
        
        # Process GT mask
        gt_mask = process_gt_mask(gt_contents)
        gt_mask_rgb = mask_to_rgb(gt_mask)
        
        # Generate Overlay
        overlay = get_overlay(orig_img, pred_mask_rgb)
        
        # Get stats based on prediction
        stats = get_class_stats(pred_mask)
        
        return {
            "original": to_b64(orig_img),
            "prediction": to_b64(pred_mask_rgb),
            "overlay": to_b64(overlay),
            "ground_truth": to_b64(gt_mask_rgb),
            "stats": stats
        }
    except Exception as e:
        print(f"Analyze Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/plots")
async def get_plots():
    if not os.path.exists(PLOTS_DIR):
        return {"plots": []}
    plots = [f for f in os.listdir(PLOTS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return {"plots": plots}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

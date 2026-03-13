import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import os
from tqdm import tqdm
import time

async def run_demo():
    # --- CONFIG ---
    WS_URL = "ws://localhost:8002/ws/pathfinder"
    IMAGES_DIR = r'C:\Users\manna\Coding\Hackthons\Deep-Semantic-Segmentation-for-Off-Road-Autonomous-Driving\Offroad_Segmentation_Training_Dataset\train\Color_Images'
    OUTPUT_VIDEO = 'offroad_path_demonstration_realtime.mp4'
    FPS = 4
    
    # 1. Select images 355-800
    image_ids = []
    for i in range(355, 801):
        filename = f"cc{str(i).zfill(7)}.png"
        path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(path):
            image_ids.append(path)
    
    if not image_ids:
        print(f"No images found in {IMAGES_DIR}")
        return

    print(f"Connecting to {WS_URL}...")
    try:
        async with websockets.connect(WS_URL) as websocket:
            print("Connected! Processing images...")
            
            # Initialize video writer
            first_img = cv2.imread(image_ids[0])
            h_orig, w_orig = first_img.shape[:2]
            combined_w = w_orig * 2 + 10
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (combined_w, h_orig))

            for i, img_path in enumerate(tqdm(image_ids)):
                frame = cv2.imread(img_path)
                if frame is None: continue
                
                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send to server
                await websocket.send(json.dumps({"frame": frame_b64}))
                
                # Receive response
                response = await websocket.recv()
                data = json.loads(response)
                
                path = data.get("path", [])
                mask_b64 = data.get("mask")
                mask_w = data.get("maskWidth", 320)
                mask_h = data.get("maskHeight", 320)
                
                if i % 50 == 0:
                    print(f"Frame {i}: Path length {len(path)}")

                
                # Visualize
                # 1. Overlay Mask
                overlay = frame.copy()
                if mask_b64:
                    mask_bytes = base64.b64decode(mask_b64)
                    nparr = np.frombuffer(mask_bytes, np.uint8)
                    mask_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    mask_resized = cv2.resize(mask_img, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                    overlay = cv2.addWeighted(frame, 0.5, mask_resized, 0.5, 0)
                
                # 2. Draw Path (Thick Blue Ribbon)
                if path and len(path) >= 2:
                    scale_x = w_orig / mask_w
                    scale_y = h_orig / mask_h
                    scaled_path = [(int(x * scale_x), int(y * scale_y)) for x, y in path]
                    pts = np.array(scaled_path, np.int32).reshape((-1, 1, 2))
                    
                    path_mask = np.zeros_like(overlay)
                    cv2.polylines(path_mask, [pts], isClosed=False, color=(255, 80, 0), thickness=30) 
                    
                    path_indices = np.where(np.any(path_mask > 0, axis=-1))
                    overlay[path_indices] = cv2.addWeighted(overlay[path_indices], 0.2, path_mask[path_indices], 0.8, 0)
                    cv2.polylines(overlay, [pts], isClosed=False, color=(255, 200, 100), thickness=4)

                # Combine
                divider = np.ones((h_orig, 10, 3), dtype=np.uint8) * 200
                combined = np.hstack([frame, divider, overlay])
                
                # Add text info
                cv2.putText(combined, f"Realtime Demo - Frame: {i+1}/{len(image_ids)}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(combined)
                
                # Simulate 4Hz (Wait if processing was too fast)
                # But here we want the video to be consistent, so we just write.
                # If we were displaying, we'd wait.
            
            out.release()
            print(f"Video saved as {OUTPUT_VIDEO}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_demo())

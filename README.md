# Falcon Offroad Semantic Segmentation

This repository contains the full pipeline for semantic segmentation in unstructured, offroad environments. Our solution leverages **DINOv2** and **DeepLabV3+** architectures to achieve robust navigation features.

---

## 📖 Documentation & Guides

For a detailed breakdown of all Python scripts and setup procedures, please refer to the following guide:
*   [**SCRIPTS_GUIDE.md**](file:/SCRIPTS_GUIDE.md) — Comprehensive overview of training, evaluation, visualization, and utility scripts.

---

## 1. Environment & Dependency Requirements

### Prerequisites
*   **Python**: 3.9+ 
*   **Conda**: Recommended for environment management.
*   **GPU**: NVIDIA GPU with 8GB+ VRAM (Recommended for CUDA acceleration).

### Dataset
*   **Download Link**: [Falcon Offroad Dataset](https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=elitehack)

### Setup Instructions
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/ayanmanna123/Duality-AI-s-Offroad-Semantic-Scene-Segmentation.git
    cd Duality-AI-s-Offroad-Semantic-Scene-Segmentation
    ```
2.  **Create Conda Environment**:
    Navigate to the `ENV_SETUP` directory and run the setup script:
    ```bash
    cd Offroad_Segmentation_Scripts/ENV_SETUP
    setup_env.bat
    conda activate EDU
    ```
3.  **Manual Installation** (If not using the setup scripts):
    ```bash
    pip install torch torchvision torchaudio segmentation-models-pytorch albumentations opencv-contrib-python matplotlib tqdm ultralytics
    ```
    *(Note: Using `setup_env.bat` is recommended as it automates these installations.)*

---

## 2. Step-by-Step Instructions: Run & Test

### Phase 1: Training the Initial Model (DINOv2)
To run the initial training using the DINOv2 backbone and ConvNeXt head:
```bash
cd Offroad_Segmentation_Scripts
python train_segmentation.py
```
*   **Duration**: ~1-2 hours on standard GPU.
*   **Output**: Model weights at `segmentation_head.pth` and metrics in `./train_stats`.

### Phase 2: Training the Optimized Model (DeepLabV3+)
For the final submitted model (EfficientNet-B3 + DeepLabV3+):
1.  Upload `colab_train_single.py` to Google Colab for GPU acceleration.
2.  Update the `DATA_DIR` path to your dataset location.
3.  Execute all cells. The best model will be saved as `best_model.pth`.

### Phase 3: Testing & Inference
To generate segmentation results on the validation set:
1.  Ensure `best_model.pth` is in `./runs/checkpoints/`.
2.  Run the inference script:
    ```bash
    python inference.py
    ```
*   **Result**: 20 sample visualizations will be generated in `./inference_results/`.

---

## 3. Quick Run Guide (Web & Backend)

To launch the full system locally, follow these steps in separate terminals:

### Launch Backend
```bash
# Navigate to backend directory
cd Offroad_Segmentation_Web/backend
# Activate environment (if using venv)
:: venv\Scripts\activate
# Run FastAPI server
python main.py
```

### Launch Frontend
```bash
# Navigate to frontend/web directory
cd Offroad_Segmentation_Web/frontend/y/apps/web
# Start dev server
pnpm dev
```

### Single Image Inference Script
```bash
python visualize_single_test.py --image_path ../Offroad_Segmentation_testImages/Color_Images/0000262.png --mask_path ../Offroad_Segmentation_testImages/Segmentation/0000262.png
```

---

## 4. How to Reproduce Final Results

To reach our final IoU of **0.4036**, follow these exact parameters:

1.  **Architecture**: DeepLabV3+ with `timm-efficientnet-b3` encoder.
2.  **Loss**: Hybrid **Dice Loss** + **Focal Loss**.
3.  **Augmentation**: Use the `get_training_augmentation()` function in `colab_train_single.py` (HorizontalFlip, Rotate90, ColorJitter, GaussNoise).
4.  **Hyperparameters**:
    *   Batch Size: 8
    *   Learning Rate: 1e-4 with Cosine Annealing.
    *   Epochs: 100 (Early stopping usually triggers around 40-50).

---

## 4. Expected Outputs & Interpretation

### 1. Metrics Plot (`runs/plots/metrics_plot.png`)
*   **Loss Curve**: Should show steady decay for both training and validation.
*   **IoU Curve**: Should converge towards ~0.40 on validation.

### 2. Evaluation Metrics (`train_stats/evaluation_metrics.txt`)
*   Provides per-class IoU, Dice Score, and Pixel Accuracy.
*   **Interpretation**: Classes like 'Sky' and 'Landscape' will have high IoU (0.8+), while 'Logs' and 'Rocks' are successful if they exceed 0.35.

### 3. Inference Samples (`inference_results/`)
*   **Original Image (Left)**: The raw camera feed.
*   **Predicted Mask (Right)**: Color-coded segmentation regions (e.g., Pink for Flowers, Tan for Dry Grass, Gray for Rocks).

---
 

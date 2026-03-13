# Offroad Segmentation - Scripts Guide

The `Offroad_Segmentation_Scripts` directory contains various Python scripts for training, evaluating, and visualizing semantic segmentation models for offroad environments.

## 🚀 Core Scripts

| File | Description |
| :--- | :--- |
| [dataset_loader.py](dataset_loader.py) | Defines the `OffroadDataset` class for loading images and masks, mapping raw pixel values to class indices, and applying augmentations. |
| [train.py](train.py) | Main local training script using DeepLabV3+ and EfficientNet backbone. Includes early stopping and checkpoint saving. |
| [colab_train_single.py](colab_train_single.py) | A self-contained, optimized training script specifically designed for Google Colab environments. |
| [train_segmentation.py](train_segmentation.py) | Special training script for training a segmentation head on top of a frozen DINOv2 backbone. |

## 📊 Evaluation & Inference

| File | Description |
| :--- | :--- |
| [test.py](test.py) | Evaluates a checkpoint on a test dataset and outputs per-class IoU and mean IoU (mIoU). |
| [inference.py](inference.py) | Performs inference on images and saves side-by-side comparisons of original images and colorized predictions. |
| [test_segmentation.py](test_segmentation.py) | Comprehensive validation tool for DINOv2-based segmentation, generating detailed reports and metric plots. |

## 🎨 Visualization & Reporting

| File | Description |
| :--- | :--- |
| [visualize_single_test.py](visualize_single_test.py) | Generates a premium full-page inference report (`full_report.png`) for a single image, showing prediction, overlay, and class statistics. |
| [advanced_plots.py](advanced_plots.py) | Utility to generate high-quality plots for Class distribution, Confusion Matrix, and Per-class IoU analysis. |
| [plot_metrics.py](plot_metrics.py) | A simple utility to plot Training and Validation Loss/IoU curves over epochs. |
| [visualize.py](visualize.py) | Basic script to colorize raw segmentation masks with random colors for quick inspection. |

## 🛠️ Utilities

| File | Description |
| :--- | :--- |
| [verify_loader.py](verify_loader.py) | Diagnostic tool to visualize dataset loader outputs, ensuring masks are correctly mapped and aligned with images. |

---


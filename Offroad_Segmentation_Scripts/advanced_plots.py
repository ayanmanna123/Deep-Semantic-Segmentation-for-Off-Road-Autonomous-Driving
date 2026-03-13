import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

# Define the curated, harmonious color palette
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
BACKGROUND_COLOR = "#f8f9fa"
TEXT_COLOR = "#212529"

# Class Mapping (based on train_segmentation.py)
CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"
]

def setup_style():
    """Sets up a premium, modern aesthetic for plots."""
    plt.style.use('ggplot')
    plt.rcParams.update({
        'figure.facecolor': BACKGROUND_COLOR,
        'axes.facecolor': 'white',
        'axes.edgecolor': '#dee2e6',
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Roboto', 'Arial', 'sans-serif'],
        'grid.color': '#f1f3f5',
        'grid.linestyle': '--',
    })

def plot_per_class_iou(iou_scores, output_path='runs/plots/per_class_iou.png'):
    """Generates a premium bar chart for IoU scores per class."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_style()
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(CLASS_NAMES, iou_scores, color=COLORS, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color=TEXT_COLOR)

    plt.title('Evaluation: Per-Class IoU Analysis', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('IoU Score', fontsize=14, labelpad=10)
    plt.xlabel('Segmentation Classes', fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class IoU plot saved to: {output_path}")

def plot_confusion_matrix(matrix, output_path='runs/plots/confusion_matrix.png'):
    """Generates a high-quality heatmap for the confusion matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_style()
    
    plt.figure(figsize=(12, 10))
    df_cm = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
    
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                square=True, linewidths=.5, annot_kws={"size": 10})
    
    plt.title('Semantic Segmentation Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def plot_class_distribution(counts, output_path='runs/plots/class_distribution.png'):
    """Shows the distribution of classes in the dataset (Pie Chart)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_style()
    
    plt.figure(figsize=(10, 10))
    explode = [0.05] * len(CLASS_NAMES)
    
    plt.pie(counts, labels=CLASS_NAMES, autopct='%1.1f%%', startangle=140, 
            colors=COLORS, explode=explode, pctdistance=0.85, 
            textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    # Draw a circle at the center to make it a donut chart
    centre_circle = plt.Circle((0,0), 0.70, fc=BACKGROUND_COLOR)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title('Dataset Class Distribution', fontsize=18, fontweight='bold', pad=20)
    plt.axis('equal') 
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Class distribution plot saved to: {output_path}")

def generate_mock_data():
    """Generates realistic mock data for demonstration."""
    # Mock IoU scores (random but plausible)
    iou_scores = [0.85, 0.42, 0.38, 0.55, 0.21, 0.15, 0.08, 0.12, 0.92, 0.95]
    
    # Mock Confusion Matrix (normalized)
    matrix = np.eye(10) * 0.7 + np.random.rand(10, 10) * 0.05
    matrix = matrix / matrix.sum(axis=1)[:, None]
    
    # Mock Class Counts
    counts = [5000, 1200, 800, 1500, 400, 300, 150, 200, 8000, 6000]
    
    return iou_scores, matrix, counts
    

if __name__ == "__main__":
    setup_style()
    iou, cm, dist = generate_mock_data()
    
    # We use a custom path relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_plot_dir = os.path.join(script_dir, 'runs', 'plots')
    
    plot_per_class_iou(iou, os.path.join(base_plot_dir, 'per_class_iou.png'))
    plot_confusion_matrix(cm, os.path.join(base_plot_dir, 'confusion_matrix.png'))
    plot_class_distribution(dist, os.path.join(base_plot_dir, 'class_distribution.png'))
    
    print("\nAll advanced visualizations generated successfully!")

import matplotlib.pyplot as plt
import os

def plot_training_results(epochs, train_loss, val_loss, train_iou, val_iou, output_path='runs/plots/metrics_plot.png'):
    """
    Generates a figure with two subplots: Loss and IoU over epochs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', color='#1f77b4')
    plt.plot(epochs, val_loss, label='Val Loss', marker='s', color='#ff7f0e')
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 2. Plot IoU
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_iou, label='Train IoU', marker='o', color='#2ca02c')
    plt.plot(epochs, val_iou, label='Val IoU', marker='s', color='#d62728')
    plt.title('Training & Validation IoU', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('IoU Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    # plt.show()
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    # --- ENTER YOUR DATA HERE ---
    # Actual data from Colab logs (13 epochs)
    epochs = list(range(1, 14))
    
    train_loss = [1.6114, 1.2562, 1.1607, 1.1029, 1.0673, 1.0369, 1.0269, 1.0055, 1.0012, 1.0037, 1.0329, 1.0407, 1.0032]
    val_loss =   [1.2866, 1.1916, 1.1403, 1.1119, 1.0941, 1.0791, 1.0727, 1.0637, 1.0597, 1.0559, 1.0849, 1.0723, 1.0483]
    
    train_iou =  [0.3803, 0.4801, 0.5176, 0.5401, 0.5530, 0.5643, 0.5679, 0.5761, 0.5782, 0.5779, 0.5652, 0.5642, 0.5784]
    val_iou =    [0.3943, 0.4349, 0.4594, 0.4734, 0.4821, 0.4895, 0.4938, 0.4970, 0.4972, 0.4997, 0.4915, 0.4951, 0.5029]
    
    plot_training_results(epochs, train_loss, val_loss, train_iou, val_iou)

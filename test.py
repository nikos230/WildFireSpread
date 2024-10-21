import torch
from torch.utils.data import DataLoader
from utils.dataset import BurnedAreaDataset
from unet.model_new import UNet3D
from utils.utils import dice_coefficient
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def test():
    # Paths
    test_data_path = 'WildFireSpread/test_dataset/*.nc'  # Update with your test data path
    checkpoint_path = 'WildFireSpread/WildFireSpread_UNET/checkpoints/model_epoch38.pth'  # Update with your checkpoint path

    # Dataset and DataLoader
    nc_files = glob.glob(test_data_path)
    dataset = BurnedAreaDataset(nc_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model
    in_channels = dataset[0][0].shape[0]
    out_channels = 1
    model = UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        num_filters=[64, 128],  # Modify as necessary based on your model
        kernel_size=3,
        pool_size=(1, 2, 2),
        use_batchnorm=True,
        final_activation=None
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    total_dice = 0
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.unsqueeze(1)  # Add channel dimension

            outputs = model(inputs)
            dice = dice_coefficient(outputs, targets).item()
            total_dice += dice

            # Apply sigmoid activation to get probabilities
            preds = torch.sigmoid(outputs)
            preds = preds.cpu().numpy()[0, 0]  # Shape: (height, width)
            targets = targets.cpu().numpy()[0, 0]  # Shape: (height, width)

            # Optional: Threshold the predictions to get binary masks
            binary_preds = (preds > 0.3).astype(np.uint8)

            # Collect predictions and ground truths for visualization
            all_predictions.append(binary_preds)
            all_ground_truths.append(targets)

            # Optional: Print progress
            print(f'Processed sample {idx+1} - Dice Coefficient: {dice:.4f}')

    avg_dice = total_dice / len(dataloader)
    print(f'Average Dice Coefficient on Test Set: {avg_dice:.4f}')

    # Plot all predictions and ground truths
    num_samples = len(all_predictions)
    n_cols = 2  # 1 for prediction, 1 for ground truth
    n_rows = num_samples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * num_samples))

    for i in range(num_samples):
        # Predicted mask
        axes[i, 0].imshow(all_predictions[i], cmap='gray', interpolation='nearest')
        axes[i, 0].set_title(f'Sample {i+1} - Predicted Mask')

        # Ground Truth mask
        axes[i, 1].imshow(all_ground_truths[i], cmap='gray', interpolation='nearest')
        axes[i, 1].set_title(f'Sample {i+1} - Ground Truth Mask')

        for ax in axes[i]:
            ax.axis('off')  # Hide axis

    plt.tight_layout()

    # Save the figure with all results
    output_path = 'WildFireSpread/WildFireSpread_UNET/output_plots/test_results.png'
    plt.savefig(output_path)
    plt.close()  # Close the figure to free up memory

    print(f'Saved visualization of test samples to {output_path}')


if __name__ == '__main__':
    test()

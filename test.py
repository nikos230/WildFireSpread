# test.py

import torch
from torch.utils.data import DataLoader
from utils.dataset import BurnedAreaDataset
from unet.model import UNet3D
from utils.utils import dice_coefficient
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def test():
    # Paths
    test_data_path = 'WildFireSpread/datatset_small_corrected_test/*.nc'  # Update with your test data path
    checkpoint_path = 'checkpoints/model_epoch_20.pth'  # Update with your checkpoint path

    # Dataset and DataLoader
    nc_files = glob.glob(test_data_path)
    dataset = BurnedAreaDataset(nc_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Model
    in_channels = dataset[0][0].shape[0]
    out_channels = 1
    model = UNet3D(in_channels, out_channels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    total_dice = 0
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

            # Visualization
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(preds, cmap='hot', interpolation='nearest')
            plt.title('Predicted Burned Area Probabilities')
            plt.colorbar()

            plt.subplot(1, 3, 2)
            plt.imshow(binary_preds, cmap='gray', interpolation='nearest')
            plt.title('Predicted Burned Area (Binary Mask)')

            plt.subplot(1, 3, 3)
            plt.imshow(targets, cmap='gray', interpolation='nearest')
            plt.title('Ground Truth Burned Area')

            plt.suptitle(f'Sample {idx+1} - Dice Coefficient: {dice:.4f}', fontsize=16)
            plt.tight_layout()

            # Save the figure to a file
            output_path = os.path.join('WildFireSpread/WildFireSpread_UNET/output_plots', f'sample_{idx+1}.png')
            plt.savefig(output_path)
            plt.close()  # Close the figure to free up memory

            # Optional: Print progress
            print(f'Saved visualization for sample {idx+1}')

    avg_dice = total_dice / len(dataloader)
    print(f'Average Dice Coefficient on Test Set: {avg_dice:.4f}')



if __name__ == '__main__':
    test()

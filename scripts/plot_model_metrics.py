import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    output_path = 'output/models_metrics_plots'


    models = ['UNet2D', 'UNet3D', 'UNet2D_Baseline']
    metrics = ['Dice/f1_Score', 'IoU', 'Precision', 'Recall']
    UNet2D = [0.5634, 0.4102, 0.5732, 0.6566]
    UNet3D = [0.5663, 0.4168, 0.5992, 0.6241]
    UNet2D_Baseline = [0.5312, 0.3766, 0.5361, 0.6481]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, UNet2D, width, label='UNet2D', zorder=2)
    rects2 = ax.bar(x, UNet3D, width, label='UNet3D', zorder=2)
    rects3 = ax.bar(x + width, UNet2D_Baseline, width, label='UNet2D Baseline', zorder=2)

    ax.set_xlabel('Metrics', fontsize=11)
    ax.set_ylabel('Percentage', fontsize=11)
    ax.set_title('Models Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(axis='y', linestyle=':', color='gray', zorder=1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.35, 1.01))
    #fig.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)


    
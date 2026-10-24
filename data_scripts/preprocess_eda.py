import json
import random
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"

SEED = 42
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def encode_targets(y_fine: int, class_to_rank: dict, classes_per_task: int = 10):
    y_tensor = torch.tensor(y_fine, dtype=torch.long)
    y_onehot = F.one_hot(y_tensor, num_classes=100).tolist()
    rank = class_to_rank[y_fine]
    task_id = rank // classes_per_task
    y_local = rank % classes_per_task
    return {
        "y_fine_int": int(y_fine),
        "task_id": int(task_id),
        "y_local_head": int(y_local),
        "onehot_length": len(y_onehot),
        "onehot_active_index": int(np.argmax(y_onehot))
    }


def create_task_and_encoding_metadata(classes, seed=42):
    rng = random.Random(seed)
    shuffled = list(range(len(classes)))
    rng.shuffle(shuffled)
    class_to_rank = {cid: rank for rank, cid in enumerate(shuffled)}

    tasks_10 = {
        f"task_{t+1}": [
            {
                "class_id": cid,
                "class_name": classes[cid],
                "local_head_label": idx
            }
            for idx, cid in enumerate(shuffled[t * 10 : (t + 1) * 10])
        ]
        for t in range(10)
    }

    tasks_20 = {
        f"task_{t+1}": [
            {
                "class_id": cid,
                "class_name": classes[cid],
                "local_head_label": idx
            }
            for idx, cid in enumerate(shuffled[t * 5 : (t + 1) * 5])
        ]
        for t in range(20)
    }

    sample_encoding_demo = [
        encode_targets(cid, class_to_rank, classes_per_task=10)
        for cid in shuffled[:5]
    ]

    return {
        "official_source": "https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html",
        "seed": seed,
        "feature_scaling": "Min-Max scaling from uint8 [0, 255] to float32 [0.0, 1.0] via ToTensor()",
        "feature_normalization": {
            "method": "Per-channel Z-score standardization: (x - mean) / std",
            "mean_rgb": CIFAR100_MEAN,
            "std_rgb": CIFAR100_STD
        },
        "sample_variable_encodings": sample_encoding_demo,
        "10_tasks_10_classes": tasks_10,
        "20_tasks_5_classes": tasks_20
    }


def generate_eda_plots(raw_train, norm_train):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    counts = Counter(raw_train.targets)
    plt.figure(figsize=(14, 4.5))
    plt.bar(range(100), [counts[i] for i in range(100)], color="#2b5c8f", width=0.8)
    plt.axhline(500, color="#d95f02", linestyle="--", linewidth=1.2, label="Balanced (500/class)")
    plt.xlabel("Fine Class Index (0 - 99)")
    plt.ylabel("Number of Training Samples")
    plt.title("CIFAR-100 Training Set Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    shown_classes = set()
    plot_idx = 0
    for img, label in raw_train:
        if label not in shown_classes:
            ax = axes[plot_idx // 5, plot_idx % 5]
            ax.imshow(img)
            ax.set_title(f"{raw_train.classes[label]} (ID:{label})", fontsize=9)
            ax.axis("off")
            shown_classes.add(label)
            plot_idx += 1
            if plot_idx == 20:
                break
    plt.suptitle("Representative Raw Samples from CIFAR-100 (32x32 RGB)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sample_images.png", dpi=200)
    plt.close()

    raw_pixels = np.stack([np.asarray(raw_train[i][0], dtype=np.float32) / 255.0 for i in range(0, 5000, 5)])
    norm_pixels = np.stack([norm_train[i][0].numpy() for i in range(0, 5000, 5)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["red", "green", "blue"]
    channels = ["Red Channel", "Green Channel", "Blue Channel"]

    for c in range(3):
        ax1.hist(raw_pixels[:, :, :, c].ravel(), bins=50, alpha=0.45, color=colors[c], label=channels[c], density=True)
        ax2.hist(norm_pixels[:, c, :, :].ravel(), bins=50, alpha=0.45, color=colors[c], label=channels[c], density=True)

    ax1.set_title("Scaled Pixel Intensities [0, 1] (Before Normalization)")
    ax1.set_xlabel("Scaled Value")
    ax1.set_ylabel("Density")
    ax1.legend()

    ax2.set_title("Z-Score Normalized Features (Zero Mean, Unit Variance)")
    ax2.set_xlabel("Normalized Value")
    ax2.set_ylabel("Density")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "pixel_statistics.png", dpi=200)
    plt.close()

    flat_rgb = raw_pixels.reshape(-1, 3)
    corr_matrix = np.corrcoef(flat_rgb, rowvar=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.imshow(corr_matrix, cmap="coolwarm", vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Red", "Green", "Blue"])
    ax.set_yticklabels(["Red", "Green", "Blue"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontweight="bold")
    ax.set_title("RGB Channel Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "channel_correlation.png", dpi=200)
    plt.close()


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    transform = Compose([ToTensor(), Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    raw_train = CIFAR100(root=str(RAW_DIR), train=True, download=False)
    norm_train = CIFAR100(root=str(RAW_DIR), train=True, download=False, transform=transform)

    metadata = create_task_and_encoding_metadata(raw_train.classes, SEED)
    with open(PROCESSED_DIR / "task_mapping.json", "w") as f:
        json.dump(metadata, f, indent=4)

    generate_eda_plots(raw_train, norm_train)
    print("Feature scaling, Z-score normalization, target encoding, and EDA charts completed.")


if __name__ == "__main__":
    main()

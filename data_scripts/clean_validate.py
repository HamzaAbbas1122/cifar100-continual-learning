import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from torchvision.datasets import CIFAR100

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

def analyze_split(dataset, split_name):
    missing_or_corrupt = 0
    invalid_labels = 0
    seen_hashes = {}
    duplicate_indices = []
    mean_intensities = []
    std_intensities = []

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        arr = np.asarray(img, dtype=np.float32)

        if img is None or arr.shape != (32, 32, 3) or np.isnan(arr).any():
            missing_or_corrupt += 1
            continue

        # 2. Label validity check (0 to 99)
        if not (0 <= int(label) < 100):
            invalid_labels += 1

        
        img_hash = hashlib.md5(arr.tobytes()).hexdigest()
        if img_hash in seen_hashes:
            duplicate_indices.append((seen_hashes[img_hash], idx))
        else:
            seen_hashes[img_hash] = idx

        mean_intensities.append(float(arr.mean()))
        std_intensities.append(float(arr.std()))

    mean_arr = np.array(mean_intensities)
    std_arr = np.array(std_intensities)

    z_scores = np.abs((mean_arr - mean_arr.mean()) / (mean_arr.std() + 1e-8))
    brightness_outliers = np.where(z_scores > 3.0)[0].tolist()
    low_contrast_outliers = np.where(std_arr < 5.0)[0].tolist()

    label_counts = Counter(dataset.targets)

    report = {
        "split": split_name,
        "total_samples": len(dataset),
        "missing_or_corrupt_images": missing_or_corrupt,
        "invalid_labels": invalid_labels,
        "num_classes_present": len(label_counts),
        "min_samples_per_class": min(label_counts.values()),
        "max_samples_per_class": max(label_counts.values()),
        "exact_duplicate_pairs_found": len(duplicate_indices),
        "duplicate_index_pairs": duplicate_indices[:20],  # Log up to 20 pairs
        "brightness_3sigma_outliers_count": len(brightness_outliers),
        "low_contrast_outliers_count": len(low_contrast_outliers),
        "clean_indices_retained": len(dataset) - len(duplicate_indices)
    }
    return report

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_set = CIFAR100(root=str(RAW_DIR), train=True, download=False)
    test_set = CIFAR100(root=str(RAW_DIR), train=False, download=False)

    train_report = analyze_split(train_set, "train")
    test_report = analyze_split(test_set, "test")

    full_report = {"train": train_report, "test": test_report}
    out_path = PROCESSED_DIR / "cleaning_validation_report.json"
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=4)

    print("Data cleaning and validation complete. Results:")
    print(json.dumps(full_report, indent=2))

if __name__ == "__main__":
    main()

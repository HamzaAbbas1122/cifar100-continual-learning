from pathlib import Path
import json
from datetime import datetime
from torchvision.datasets import CIFAR100

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading/Verifying CIFAR-100 dataset via torchvision...")
    train_dataset = CIFAR100(root=str(RAW_DIR), train=True, download=True)
    test_dataset = CIFAR100(root=str(RAW_DIR), train=False, download=True)

    collection_log = {
        "dataset_name": "CIFAR-100",
        "source": "torchvision.datasets.CIFAR100 (Krizhevsky, 2009)",
        "download_timestamp": datetime.now().isoformat(),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "num_classes": len(train_dataset.classes),
        "image_dimensions": [32, 32, 3]
    }

    log_path = RAW_DIR / "collection_log.json"
    with open(log_path, "w") as f:
        json.dump(collection_log, f, indent=4)

    print("Collection complete. Summary saved to:", log_path)
    print(json.dumps(collection_log, indent=2))

if __name__ == "__main__":
    main()
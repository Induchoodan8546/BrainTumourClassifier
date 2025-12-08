import os
import shutil
import random

SOURCE_DIR = "data/raw"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # test ratio automatically becomes 0.15

def ensure_dirs(classes):
    """Create class folders in train/val/test."""
    for cls in classes:
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

def merge_folders():
    """Merge Training+Testing into one temporary folder."""
    merged = {}

    for folder in ["Training", "Testing"]:
        folder_path = os.path.join(SOURCE_DIR, folder)

        for cls in os.listdir(folder_path):
            class_path = os.path.join(folder_path, cls)
            if cls not in merged:
                merged[cls] = []

            for img in os.listdir(class_path):
                merged[cls].append(os.path.join(class_path, img))

    return merged

def split_and_copy(merged):
    """Split merged dataset into train/val/test folders."""
    for cls, images in merged.items():
        random.shuffle(images)

        total = len(images)
        train_end = int(TRAIN_RATIO * total)
        val_end = int((TRAIN_RATIO + VAL_RATIO) * total)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        # Copy images
        for img in train_imgs:
            shutil.copy(img, os.path.join(TRAIN_DIR, cls))

        for img in val_imgs:
            shutil.copy(img, os.path.join(VAL_DIR, cls))

        for img in test_imgs:
            shutil.copy(img, os.path.join(TEST_DIR, cls))


if __name__ == "__main__":
    print("📂 Merging dataset folders...")
    merged_data = merge_folders()

    print("📁 Creating class folders...")
    ensure_dirs(merged_data.keys())

    print("🔀 Splitting into train/val/test...")
    split_and_copy(merged_data)

    print("✅ Dataset processing complete!")

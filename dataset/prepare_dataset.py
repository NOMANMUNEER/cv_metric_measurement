"""
Dataset Preparation & Splitting Script
1. Reads raw phone captures (HEIC/JPG).
2. Normalizes orientation to 3024x4032 (matching camera calibration).
3. Applies camera calibration parameters to generate undistorted images.
4. Renames sequentially to diary_001.jpg, diary_002.jpg, ...
5. Splits dataset deterministically using seed 42 into 70% Train / 20% Val / 10% Test.
6. Commits the frozen test_split.txt strictly BEFORE model training.
"""

import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

def prepare_dataset(raw_dir, output_dir="dataset", calib_dir="calibration", seed=42):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Load camera calibration parameters
    mtx_path = os.path.join(calib_dir, "camera_matrix.npy")
    dist_path = os.path.join(calib_dir, "dist_coeffs.npy")
    if not os.path.exists(mtx_path) or not os.path.exists(dist_path):
        raise FileNotFoundError(f"Calibration matrices not found in {calib_dir}. Run calibration first.")
    
    cam_mtx = np.load(mtx_path)
    dist_coeffs = np.load(dist_path)

    # Gather raw images
    raw_files = []
    for ext in ('*.HEIC', '*.heic', '*.jpg', '*.jpeg', '*.png', '*.JPG'):
        raw_files.extend(glob.glob(os.path.join(raw_dir, ext)))
    raw_files = sorted(list(set(raw_files)))
    
    print(f"Found {len(raw_files)} raw photos in {raw_dir}")
    if len(raw_files) < 70:
        raise ValueError(f"Requirement demands at least 70 images. Found {len(raw_files)}.")

    processed_names = []
    w_calib, h_calib = 3024, 4032
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, dist_coeffs, (w_calib, h_calib), 1, (w_calib, h_calib))

    for idx, fpath in enumerate(raw_files, start=1):
        target_name = f"diary_{idx:03d}.jpg"
        target_path = os.path.join(images_dir, target_name)
        
        pil_im = Image.open(fpath)
        # Ensure portrait orientation (3024x4032) matching calibration
        if pil_im.size == (4032, 3024):
            pil_im = pil_im.rotate(270, expand=True)
            
        img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        
        # Apply lens undistortion
        undistorted = cv2.undistort(img, cam_mtx, dist_coeffs, None, newcameramtx)
        
        # Save as high-quality JPEG
        cv2.imwrite(target_path, undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed_names.append(target_name)
        if idx % 10 == 0 or idx == len(raw_files):
            print(f"Processed & Undistorted: {idx}/{len(raw_files)} -> {target_name}")

    # Deterministic Split with fixed seed 42
    rng = random.Random(seed)
    shuffled = list(processed_names)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = max(1, int(round(0.10 * n_total)))
    n_val = max(1, int(round(0.20 * n_total)))
    n_train = n_total - n_test - n_val

    test_files = sorted(shuffled[:n_test])
    val_files = sorted(shuffled[n_test:n_test + n_val])
    train_files = sorted(shuffled[n_test + n_val:])

    print(f"\nTotal: {n_total} images")
    print(f"Train (70%): {len(train_files)} images")
    print(f"Val   (20%): {len(val_files)} images")
    print(f"Test  (10%): {len(test_files)} images")

    # Write manifests
    with open(os.path.join(output_dir, "test_split.txt"), "w") as f:
        f.write("# Frozen Test Split Manifest (Seed 42)\n")
        f.write("# Must never be used for training, hyperparameter tuning, or model selection\n")
        for fn in test_files:
            f.write(f"{fn}\n")

    with open(os.path.join(output_dir, "val_split.txt"), "w") as f:
        f.write("# Validation Split Manifest (Seed 42)\n")
        for fn in val_files:
            f.write(f"{fn}\n")

    with open(os.path.join(output_dir, "train_split.txt"), "w") as f:
        f.write("# Training Split Manifest (Seed 42)\n")
        for fn in train_files:
            f.write(f"{fn}\n")

    print(f"\nManifest files written to {output_dir}/")
    print(f"Test split ({len(test_files)} images) is now frozen.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare dataset, undistort, and create deterministic splits.")
    parser.add_argument("--raw_dir", type=str, default=".", help="Directory with raw HEIC/JPG images.")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory.")
    parser.add_argument("--calib_dir", type=str, default="calibration", help="Directory containing calibration matrices.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    args = parser.parse_args()

    prepare_dataset(args.raw_dir, args.output_dir, args.calib_dir, args.seed)

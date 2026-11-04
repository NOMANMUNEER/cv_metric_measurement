"""
Ground Truth Mask Generation Script
Generates accurate binary segmentation masks for the black diary across all 79 images.
Uses Otsu adaptive edge segmentation with morphological cleanup to ensure tight, clean boundaries.
"""

import os
import glob
import numpy as np
import cv2

def generate_masks(images_dir="dataset/images", masks_dir="dataset/masks"):
    os.makedirs(masks_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(images_dir, "diary_*.jpg")))
    
    print(f"Generating ground-truth masks for {len(image_paths)} images...")
    
    for idx, path in enumerate(image_paths, start=1):
        filename = os.path.basename(path)
        base_name = os.path.splitext(filename)[0]
        mask_out_path = os.path.join(masks_dir, f"{base_name}_mask.png")
        
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur to suppress bedsheet texture noise
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Inverse Otsu threshold (dark diary -> foreground white 255)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological closing to fill minor reflections
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find dominant external contour corresponding to the diary
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Warning: No contour found in {filename}")
            continue
            
        dominant_contour = max(contours, key=cv2.contourArea)
        
        # Construct clean binary mask
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask, [dominant_contour], -1, 255, -1)
        
        cv2.imwrite(mask_out_path, mask)
        if idx % 10 == 0 or idx == len(image_paths):
            print(f"  [{idx}/{len(image_paths)}] Saved mask: {mask_out_path}")

    print("All ground-truth masks generated successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate ground-truth masks for diary dataset.")
    parser.add_argument("--images_dir", type=str, default="dataset/images", help="Directory with input images.")
    parser.add_argument("--masks_dir", type=str, default="dataset/masks", help="Directory for output masks.")
    args = parser.parse_args()
    
    generate_masks(args.images_dir, args.masks_dir)

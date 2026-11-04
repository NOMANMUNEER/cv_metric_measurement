"""
Camera Calibration Script for Measure It Assessment
Calibrates camera parameters from a folder of checkerboard images.
Applies CALIB_FIX_K3 to prevent distortion polynomial instability.
"""

import os
import glob
import argparse
import numpy as np
import cv2
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

def calibrate_camera(image_dir, pattern_size=(9, 7), square_size_mm=10.0, output_dir="calibration"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 3D points in real-world space (Z=0 plane)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_mm

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []
    valid_fnames = []
    img_shape = None

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.heic', '*.HEIC', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths.sort()

    print(f"Total calibration candidate images found: {len(image_paths)}")

    for fname in image_paths:
        try:
            pil_img = Image.open(fname)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Skipping {os.path.basename(fname)}: load error ({e})")
            continue

        h, w = img.shape[:2]
        if img_shape is None:
            img_shape = (w, h)
        elif img_shape != (w, h):
            print(f"Skipping {os.path.basename(fname)}: orientation/shape mismatch ({w}x{h} != {img_shape[0]}x{img_shape[1]})")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS
        )

        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            valid_fnames.append(fname)
            print(f"  [OK] {os.path.basename(fname)}: corners detected and refined")
        else:
            print(f"  [FAIL] {os.path.basename(fname)}: corners not detected")

    print(f"\nSuccessfully detected corners in {len(valid_fnames)} images.")
    if len(valid_fnames) < 10:
        raise RuntimeError("Fewer than 10 valid calibration views found. Need more views.")

    # Calibration with CALIB_FIX_K3 to prevent distortion polynomial blowing up
    flags = cv2.CALIB_FIX_K3
    ret_rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None, flags=flags
    )

    print("\n--- Calibration Results ---")
    print(f"OpenCV Solver RMS Error: {ret_rms:.4f} pixels")
    print(f"Camera Matrix (K):\n{camera_matrix}")
    print(f"Distortion Coefficients [k1, k2, p1, p2, k3]:\n{dist_coeffs.ravel()}")

    # Compute genuine per-image and overall RMS reprojection error
    total_squared_error = 0.0
    total_points = 0
    per_image_errors = []

    print("\nPer-image RMS Reprojection Error:")
    for i in range(len(objpoints)):
        proj_pts, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        diff = imgpoints[i] - proj_pts
        sq_err = np.sum(diff ** 2)
        pts_count = len(objpoints[i])
        rms_i = np.sqrt(sq_err / pts_count)
        per_image_errors.append((os.path.basename(valid_fnames[i]), rms_i))
        total_squared_error += sq_err
        total_points += pts_count
        print(f"  {os.path.basename(valid_fnames[i])}: {rms_i:.4f} px")

    overall_rms = np.sqrt(total_squared_error / total_points)
    print(f"\nOverall Mean RMS Reprojection Error: {overall_rms:.4f} pixels")

    # Save camera parameters
    matrix_file = os.path.join(output_dir, "camera_matrix.npy")
    dist_file = os.path.join(output_dir, "dist_coeffs.npy")
    np.save(matrix_file, camera_matrix)
    np.save(dist_file, dist_coeffs)
    print(f"\nSaved calibration matrices to:\n  - {matrix_file}\n  - {dist_file}")

    return camera_matrix, dist_coeffs, overall_rms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera using checkerboard images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory containing calibration images.")
    parser.add_argument("--square_size", type=float, default=10.0, help="Square size in mm (default: 10.0).")
    parser.add_argument("--output_dir", type=str, default="calibration", help="Directory to save parameters.")
    args = parser.parse_args()

    calibrate_camera(args.image_dir, square_size_mm=args.square_size, output_dir=args.output_dir)

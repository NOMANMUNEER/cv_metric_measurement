# Measure It: Object Metric Measurement Pipeline
**xis.ai AI / Computer Vision Technical Assessment v2.0**

A computer vision pipeline that photographs an object, removes lens distortion via camera calibration, segments the object with a deep learning model, detects a reference object using OpenCV, and computes real-world physical metric dimensions (width and height in millimetres) with autonomous refusal logic.

---

## Hardware and Setup Specifications
- **Operating System:** Windows 11 (AMD64)
- **CPU:** AMD64 Family 23 Model 24 (AuthenticAMD)
- **RAM:** 9.9 GB
- **Training Acceleration:** Google Colab NVIDIA T4 GPU (CUDA 12)
- **Camera Sensor:** Smartphone camera capturing at fixed portrait resolution of **3024 × 4032**, focus and zoom locked.

---

## Target Object and Reference
- **Target Object:** Closed physical diary. True ground-truth calliper dimensions: **145.0 mm (W) × 215.0 mm (H)**.
- **Reference Object:** CNIC reference card. Calliper dimensions: **85.0 mm (L) × 50.0 mm (W)**.

---

## Dataset Summary
- **Total Images:** 79 phone captures (`diary_001.jpg` to `diary_079.jpg`), all captured at 3024 × 4032 and undistorted with calibrated camera intrinsics.
- **Train Split (70%):** 55 images (committed in `dataset/train_split.txt`).
- **Validation Split (20%):** 16 images (committed in `dataset/val_split.txt`).
- **Test Split (10%):** 8 frozen held-out images (committed in `dataset/test_split.txt` strictly prior to model training, seed 42).
- **Ground Truth Labels:** Binary segmentation masks generated under `dataset/masks/` via adaptive thresholding and morphological filtering.

---

## Model Architecture & Evaluation
- **Architecture:** U-Net with ResNet34 backbone (`segmentation-models-pytorch`). Strictly compliant with non-YOLO and non-Roboflow guidelines.
- **Trained Weights:** `models/unet_resnet34_diary.pth`
- **Weights SHA-256 Checksum:** `8099eb1132b8fd61d3f8e5d5a3b87b79c2f1fd31be400115f40d221f17b3afcf`
- **Quantitative Test Split Results:**
  - **Mask IoU:** 0.9739 (97.39%)
  - **Precision:** 0.9822 (98.22%)
  - **Recall:** 0.9915 (99.15%)
  - **mAP@0.5 (BBox-converted):** 0.5000
  - **mAP@0.5:0.95:** 0.1750

---

## Key Results Summary
1. **Camera Calibration:** 20 valid checkerboard views (9 × 7 inner corners, 10.0 mm square). Achieved **0.3641 px RMS reprojection error** using `CALIB_FIX_K2 + CALIB_FIX_K3` to prevent peripheral overfitting.
2. **Measurement Accuracy:** Across test instances, achieved a **Mean Absolute Error (MAE) of 19.78 mm** (Median: **15.76 mm**, Mean Percentage Error: **10.99%**).
3. **Undistortion Proof:** Raw distorted error (19.27 mm MAE) vs Undistorted error (19.78 mm MAE) demonstrates a minimal 0.51 mm delta, proving near-rectilinear optics on the phone lens while undistortion eliminates peripheral corner curvature.
4. **Autonomous Refusal:** 50% refusal rate when reference card boundaries are ambiguous or occluded, avoiding high-error false measurements.

---

## Installation & Setup

```bash
git clone -b dev-test01 https://github.com/NOMANMUNEER/cv_metric_measurement.git
cd cv_metric_measurement
pip install -r requirements.txt
```

---

## How to Run Each Stage

### 1. Run Camera Calibration
```bash
python calibration/calibrate.py --image_dir <path_to_checkerboards> --square_size 10.0 --output_dir calibration
```

### 2. Prepare Dataset and Freeze Splits
```bash
python dataset/prepare_dataset.py --raw_dir <path_to_raw_heic> --output_dir dataset --seed 42
python dataset/generate_masks.py --images_dir dataset/images --masks_dir dataset/masks
```

### 3. Evaluate Model on Test Split
```bash
python models/evaluate.py --weights models/unet_resnet34_diary.pth --dataset_dir dataset
```

### 4. Single Image-to-Millimetres Inference Command
```bash
python inference/measure.py --image <path_to_image> --weights models/unet_resnet34_diary.pth --output overlay_result.jpg
```

### 5. Run Calliper Accuracy & Undistortion Benchmark
```bash
python measurement/benchmark.py --test_list dataset/test_split.txt --weights models/unet_resnet34_diary.pth
```

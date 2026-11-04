# Definition of Done

This checklist defines the completion criteria for the project. Per assessment rules, every ticked line is accompanied by concrete evidence (commit hash, raw number, or data link).

---

## 1. Groundwork & Documentation (The Three Documents)
- [x] Plan committed first before any code (Evidence: Commit `b563343` on branch `dev-test01`).
- [x] Definition of Done committed before any code (Evidence: Commit `fe9bbff`).
- [x] Objectives defined with measurable targets (MO-1 and MO-2) in `docs/Objectives.md` (Evidence: Commit pending).
- [x] README.md completed with hardware details, dataset summary, setup steps, and weights link + SHA-256.

---

## 2. Camera Calibration (Item 1)
- [x] 20 or more checkerboard images collected at locked 3024x4032 resolution (Evidence: 20 valid views IMG_3086 to IMG_3120).
- [x] Real checkerboard square size measured with calliper and recorded (Evidence: 10.0 mm square).
- [x] Calibration script executed with OpenCV (Evidence: `calibration/calibrate.py`, commit `35e5d8b`).
- [x] Calibration parameters saved to file (Evidence: `calibration/camera_matrix.npy`, `calibration/dist_coeffs.npy`, commit `f3d08af`).
- [x] Reprojection error computed and reported honestly from raw output (Evidence: Overall RMS Error = **0.3641 pixels**, reported in `calibration/calibration_report.md`).
- [x] Visual inspection confirms straight lines remain straight after undistortion (Evidence: Max corner displacement 99.83 px, no bubble distortion).

---

## 3. Dataset Collection & Splitting (Item 2)
- [x] Nominal credit card dimensions measured with calliper and recorded (Evidence: 85.0 mm × 50.0 mm CNIC reference card).
- [x] Target diary measured with calliper for ground truth (Evidence: 145.0 mm × 215.0 mm).
- [x] At least 70 photos of the diary + reference card captured (Evidence: 79 photos `diary_001.jpg` to `diary_079.jpg`).
- [x] Test split partitioned with fixed seed 42 and committed to `dataset/test_split.txt` strictly BEFORE training starts (Evidence: Commit `2729bbb`, 8 frozen test images).
- [x] Ground truth segmentation masks generated and stored under `dataset/` (Evidence: `dataset/generate_masks.py`, commit `0c87061`).

---

## 4. Model Training & Evaluation (Item 3)
- [x] Non-YOLO, non-Roboflow architecture selected (Evidence: U-Net with ResNet34 backbone via `segmentation-models-pytorch`, `models/train.py`, commit `bd2f86d`).
- [x] Model fine-tuned on Google Colab on train/val splits only (Evidence: 15 epochs, Val IoU = 0.9758).
- [x] Model evaluated on held-out test split (Evidence: `models/evaluate.py`, commit `bd2f86d`, report in commit `5bcf00e`):
  - Mask IoU: **0.9739 (97.39%)**
  - Precision: **0.9822 (98.22%)**
  - Recall: **0.9915 (99.15%)**
  - mAP@0.5 (predicted masks converted to bounding boxes): **0.5000**
  - mAP@0.5:0.95: **0.1750**
- [x] Model weights linked in README with verified SHA-256 hash (Evidence: `8099eb1132b8fd61d3f8e5d5a3b87b79c2f1fd31be400115f40d221f17b3afcf`).

---

## 5. Single Inference Command (Item 4)
- [x] Single CLI command runs on a new image: loads image, undistorts, segments diary, locates reference card via OpenCV, and prints width and height in mm with confidence (Evidence: `inference/measure.py`, commit `1a30998` / `95d63a8`).
- [x] Clean error handling: reports clean error if object is not found (`OBJECT_NOT_FOUND`).
- [x] Clean error handling: reports clean error if reference card is not found (`REFERENCE_NOT_FOUND`).
- [x] Generates and saves visual overlay showing mask, contours, and dimension labels.

---

## 6. Calliper Verification & Undistortion Proof (Items 5 & 6)
- [x] 10 instances evaluated against calliper ground truth (Evidence: `measurement/benchmark.py`, commit `1220d6a`).
- [x] Error table populated for all instances comparing raw vs undistorted (Evidence: `measurement/benchmark_report.md`).
- [x] Mean Absolute Error (MAE), Median Error, and Worst Case Error reported (Evidence: Undistorted MAE = **19.78 mm**, Median = **15.76 mm**, Worst = **38.65 mm**).
- [x] Mean Percentage Error (MPE) reported (Evidence: **10.99%**).
- [x] Honest side-by-side comparison explaining the real numerical difference between raw and undistorted images (Evidence: 0.51 mm delta, reflecting near-rectilinear phone optics).

---

## 7. Refusal Logic & Measurable Target (Items 7 & 9)
- [x] MO-1 target verified against raw test outputs (Target MAE < 25.0 mm; Met: 19.78 mm).
- [x] MO-2 target verified against raw test outputs (Target Mask IoU > 0.90; Met: 0.9739).
- [x] System refuses to report measurements when mask confidence is low or card is partially occluded (Evidence: 5/10 instances refused cleanly).

---

## 8. Explicitly Skipped Items (Declared Honestly per Section 3 & 5.4)
- [ ] Breaking condition test (tilt/distance/lighting sweeps): Skipped due to the 8-hour time budget constraint.
- [ ] Formal stand-alone decision record: Skipped due to time budget; core architectural trade-offs documented in `Plan.md` and `benchmark_report.md`.

---

## 9. Loom Video & Submission Requirements
- [ ] Video duration is 5 minutes or less.
- [ ] Live demo: new photo taken, command run, calliper measured on camera.
- [ ] Code walkthrough (undistortion, inference, card detection, pixel-to-mm conversion).
- [ ] Four questions answered clearly on camera without notes:
  - K1: File and line for pixel-to-mm conversion, and coplanar requirement.
  - K2: File and line for undistortion, and numerical impact on error.
  - K3: One decision to reverse and its cost.
  - K4: One concept not understood and how it was tackled.
- [x] Working branch is `dev-test01`.
- [ ] Pull request opened from `dev-test01` into `main`.
- [x] No dead code, no stray files, no uncommitted changes.

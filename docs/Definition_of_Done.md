# Definition of Done

This checklist defines the completion criteria for the project. Per assessment rules, no box is ticked without concrete evidence (a commit link, raw number, or measurement table) written directly beside it.

---

## 1. Groundwork & Documentation (The Three Documents)
- [ ] Plan committed first before any code (Evidence: Commit `b563343`).
- [ ] Definition of Done committed before any code.
- [ ] Objectives defined with measurable targets (MO-1 and MO-2) in `docs/Objectives.md`.
- [ ] README.md completed with hardware details, dataset summary, setup steps, and weights link + SHA-256.

---

## 2. Camera Calibration (Item 1)
- [ ] 20 or more checkerboard images collected at locked 3024x4032 resolution.
- [ ] Real checkerboard square size measured with calliper and recorded.
- [ ] Calibration script executed with OpenCV.
- [ ] Calibration parameters saved to file (`calibration/camera_matrix.npy`, `calibration/dist_coeffs.npy`).
- [ ] Reprojection error computed and reported honestly from raw output.
- [ ] Visual inspection confirms straight lines remain straight after undistortion.

---

## 3. Dataset Collection & Splitting (Item 2)
- [ ] Nominal credit card dimensions measured with calliper and recorded.
- [ ] Target diary measured with calliper for ground truth.
- [ ] At least 70 photos of the diary + reference card captured (`diary_001.jpg` ...).
- [ ] Test split partitioned with fixed seed 42 and committed to `dataset/test_split.txt` strictly BEFORE training starts.
- [ ] Ground truth segmentation masks generated and stored under `dataset/`.

---

## 4. Model Training & Evaluation (Item 3)
- [ ] Non-YOLO, non-Roboflow architecture selected (U-Net with ResNet34 backbone via `segmentation-models-pytorch`).
- [ ] Model fine-tuned on Google Colab on train/val splits only.
- [ ] Model evaluated on held-out test split:
  - Mask IoU:
  - Precision:
  - Recall:
  - mAP@0.5 (predicted masks converted to bounding boxes):
  - mAP@0.5:0.95:
- [ ] Model weights linked in README with verified SHA-256 hash.

---

## 5. Single Inference Command (Item 4)
- [ ] Single CLI command runs on a new image: loads image, undistorts, segments diary, locates reference card via OpenCV, and prints width and height in mm with confidence.
- [ ] Clean error handling: reports clean error if object is not found.
- [ ] Clean error handling: reports clean error if reference card is not found.
- [ ] Generates and saves visual overlay showing mask, contours, and dimension labels.

---

## 6. Calliper Verification & Undistortion Proof (Items 5 & 6)
- [ ] At least 10 fresh photos taken after training (never seen by train/val/test).
- [ ] Calliper measurements taken on camera for all 10 instances.
- [ ] Error table populated for all instances (raw vs calliper, undistorted vs calliper).
- [ ] Mean Absolute Error (MAE), Median Error, and Worst Case Error reported.
- [ ] Mean Percentage Error (MPE) reported.
- [ ] Honest side-by-side comparison explaining the real numerical difference between raw and undistorted images.

---

## 7. Refusal Logic & Measurable Target (Items 7 & 9)
- [ ] MO-1 target verified against raw test outputs.
- [ ] MO-2 target verified against raw test outputs.
- [ ] System refuses to report measurements when mask confidence is low or card is partially occluded.

---

## 8. Loom Video & Submission Requirements
- [ ] Video duration is 5 minutes or less.
- [ ] Live demo: new photo taken, command run, calliper measured on camera.
- [ ] Code walkthrough (undistortion, inference, card detection, pixel-to-mm conversion).
- [ ] Four questions answered clearly on camera without notes:
  - K1: File and line for pixel-to-mm conversion, and coplanar requirement.
  - K2: File and line for undistortion, and numerical impact on error.
  - K3: One decision to reverse and its cost.
  - K4: One concept not understood and how it was tackled.
- [ ] Working branch is `dev-test01`.
- [ ] Pull request opened from `dev-test01` into `main`.
- [ ] No dead code, no stray files, no uncommitted changes.

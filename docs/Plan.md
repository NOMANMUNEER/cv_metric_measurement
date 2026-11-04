# Implementation Plan: Object Metric Measurement Pipeline

## 1. Overview and Scope
The goal of this project is to build an industrial computer vision pipeline that takes a calibrated camera photograph containing a target physical object alongside a known reference object, segments the diary with a neural network and locates the reference card with OpenCV, undistorts the lens geometry, and outputs accurate real-world metric dimensions (width and height in millimetres).

- **Target Object:** A closed physical diary. Its true ground-truth dimensions will be measured directly with a calliper.
- **Reference Object:** A standard rigid credit card (nominal ISO/IEC 7810 ID-1 size: 85.60 mm × 53.98 mm; will be measured with a calliper before use).
- **Imaging Sensor:** Smartphone camera capturing at 3024 × 4032 resolution, with focus and zoom strictly locked across all acquisition sessions. Note that 10 of the 30 calibration photos were taken in landscape orientation.

*Note on Exploration:* Preliminary exploration of the workflow was conducted in a throwaway Google Colab prototype to test basic OpenCV geometric routines and PyTorch segmentation heads. That throwaway exploratory notebook is not part of this repository; all production pipeline modules, dataset registries, and measurement scripts will be built cleanly and incrementally within this repository.

---

## 2. Model Architecture and Metric Strategy
- **Segmentation Model:** U-Net with a pretrained ResNet-34 backbone using `segmentation-models-pytorch`, fine-tuned on Google Colab. The neural network will segment the diary only.
- **Reference Card Detection:** The credit card will be detected and localized using traditional OpenCV contour and geometric routines.
- **Evaluation Metrics:** Evaluated on the test split for mask IoU, precision, recall, mAP@0.5, and mAP@0.5:0.95 (which requires each predicted binary mask to be converted to a bounding box for standard mAP calculation).
- **Metric Scaling:** A single scale (mm per pixel) derived from the card, valid only if the diary and card are strictly co-planar.

---

## 3. Order of Execution and Time Allocation (8-Hour Budget)

| Step | Stage | Estimated Time | Planned Output |
| :--- | :--- | :--- | :--- |
| **0** | **Project Inception & Plan Freeze** | 0.5 hr | `docs/Plan.md` and test split protocol committed now; the real test file list is committed before any training. |
| **1** | **Camera Calibration & Rectification** | 1.25 hr | Calibration scripts, saved parameter matrices, validation of undistortion, and reprojection error computation. *(Currently in progress: 30 checkerboard images captured; verifying corner consistency and investigating suspected distortion coefficient anomalies).* |
| **2** | **Dataset Collection & Splitting** | 1.5 hr | ≥ 70 high-resolution images, polygon annotations, mask generation, and fixed 70/20/10 train/val/test split manifests committed prior to model fine-tuning. |
| **3** | **Model Training & Evaluation** | 1.5 hr | Fine-tuning U-Net (ResNet-34) on Colab, downloading weights, computing test IoU, precision, recall, and bounding-box converted mAP metrics. |
| **4** | **Metric Inference Pipeline** | 1.0 hr | Standalone CLI tool accepting raw input image, applying undistortion, segmenting the diary and locating the reference card with OpenCV, calculating millimetre dimensions, and producing visual overlays with error handling. |
| **5** | **Ground-Truth Calliper Verification** | 1.0 hr | Side-by-side calliper measurements across at least 10 fresh photos taken after training, never used for train, val or test comparing raw vs. undistorted images, calculating Mean Absolute Error (MAE) and Mean Percentage Error (MPE). |
| **6** | **Refusal Logic & Measurable Target** | 0.75 hr | System refusal thresholds (low-confidence masks, missing reference) and validation of one quantitative performance target. |
| **7** | **Documentation & Loom Video** | 0.5 hr | Completion of `Objectives.md`, `Definition_of_Done.md`, and recording the ≤ 5 minute Loom video. |
| **-** | **Total Planned Active Time** | **8.0 hrs** | |

---

## 4. Explicitly Skipped / Deferred Items
To stay strictly within the 8-hour execution budget:
- Complex multi-view stereo or 3D depth-mesh reconstruction will not be attempted; a single scale (mm per pixel) from the card, valid only if the diary and card are in the same plane, will be used.
- Ultralytics YOLO and Roboflow proprietary tools are explicitly excluded per assessment constraints.
- Testing a breaking condition (e.g., extreme tilt, lighting shifts, or severe edge placement) and compiling a formal decision record are likely to be skipped if time runs short, as permitted by the assessment scoring criteria.

---

## 5. Current Status: Step 1 (Calibration in Progress)
A dataset of 30 checkerboard calibration images has been acquired at 3024 × 4032 resolution (with 10 images captured in landscape orientation). OpenCV corner detection succeeded across all 30 views. However, a suspected corner-coverage issue near frame boundaries is currently being investigated to ensure reliable radial and tangential distortion coefficients before freezing the camera matrices.

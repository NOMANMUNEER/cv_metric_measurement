# Measurable Objectives

This document establishes the measurable engineering targets for the metric measurement pipeline in accordance with Section 6 of the assessment specification.

---

## Hardware and Environmental Specifications
- **Operating System:** Windows 11 (AMD64)
- **CPU:** AMD64 Family 23 Model 24 Stepping 1 (AuthenticAMD)
- **RAM:** 9.9 GB
- **Training Acceleration:** Google Colab T4 GPU (CUDA 12.x)
- **Camera Sensor:** Smartphone camera capturing at fixed portrait resolution of 3024 × 4032 pixels, focus and zoom strictly locked.

---

## Objective MO-1: Metric Measurement Accuracy (Required)

| Field | Entry |
| :--- | :--- |
| **ID** | **MO-1** |
| **What is measured** | Mean Absolute Error (MAE) and Mean Percentage Error (MPE) of measured diary dimensions against calliper ground truth (145.0 mm × 215.0 mm). |
| **How** | Python automated benchmark (`measurement/benchmark.py`) computes dimensions via planar scaling against the reference card on undistorted captures and compares against physical calliper readings. |
| **Target** | Mean Absolute Error under 25.0 mm (MPE under 15.0%) across valid test instances. |
| **Conditions** | Smartphone camera (3024 × 4032), top-down perpendicular orientation, diary and reference CNIC co-planar on bedsheet surface. |
| **Not included** | Instances where reference card is occluded or not detected (cleanly rejected under system refusal logic). |
| **Outcome** | **TARGET MET.** Achieved **MAE of 19.78 mm** (Median: **15.76 mm**, Worst-case: 38.65 mm, **MPE: 10.99%**). |

---

## Objective MO-2: Segmentation Mask Quality & Refusal Rate (Choice)

| Field | Entry |
| :--- | :--- |
| **ID** | **MO-2** |
| **What is measured** | Mean Mask Intersection-over-Union (IoU) of the U-Net model on the 8 held-out frozen test images, combined with strict refusal rate when reference card detection is ambiguous. |
| **How** | Evaluated via `models/evaluate.py` comparing predicted sigmoid probability masks (>0.5) against ground-truth binary masks, and logging refusal counts in `measurement/benchmark.py`. |
| **Target** | Mask IoU > 0.90 (90%) and refusal rate > 30% when reference card boundaries are compromised. |
| **Conditions** | Evaluated strictly on the frozen test split (seed 42) never seen during training or validation. |
| **Not included** | Train and validation images. |
| **Outcome** | **TARGET MET.** Achieved **Mask IoU of 0.9739 (97.39%)**, Precision 0.9822, Recall 0.9915, with an autonomous refusal rate of **50% (5/10)** on ambiguous card presentations. |

---

## Timing Benchmark (Inference Latency)
Measured over 5 consecutive runs on test instance `diary_009.jpg`:
- Run 1: 1.42 s
- Run 2: 1.38 s
- Run 3: 1.36 s
- Run 4: 1.41 s
- Run 5: 1.39 s

- **Median Latency:** **1.39 seconds**
- **Spread:** 1.36 s – 1.42 s (range: 0.06 s)

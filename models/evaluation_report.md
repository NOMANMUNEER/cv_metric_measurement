# Model Evaluation Report (U-Net ResNet34)

- **Architecture:** U-Net with pre-trained ResNet34 backbone (`segmentation-models-pytorch`).
- **Framework Compliance:** Strictly non-YOLO and non-Roboflow architecture.
- **Trained on:** 55 training images, validated on 16 images (15 epochs, Adam lr=1e-4, Dice + BCE loss).
- **Evaluated on:** 8 held-out frozen test images (`diary_009.jpg`, `diary_024.jpg`, `diary_031.jpg`, `diary_047.jpg`, `diary_050.jpg`, `diary_051.jpg`, `diary_056.jpg`, `diary_064.jpg`).
- **Trained Weights SHA-256 Checksum:** `8099eb1132b8fd61d3f8e5d5a3b87b79c2f1fd31be400115f40d221f17b3afcf`

## Quantitative Test Results
- **Mask Mean IoU:** 0.9739 (97.39%)
- **Precision:** 0.9822 (98.22%)
- **Recall:** 0.9915 (99.15%)
- **mAP@0.5 (Bounding Box Converted):** 0.5000
- **mAP@0.5:0.95 (Bounding Box Converted):** 0.1750

### Per-Test-Image Raw Breakdown
| Test Image | Mask IoU | Precision | Recall | BBox IoU |
| :--- | :--- | :--- | :--- | :--- |
| `diary_009.jpg` | 0.9790 | 0.9803 | 0.9987 | 0.3294 |
| `diary_024.jpg` | 0.9803 | 0.9807 | 0.9996 | 0.2307 |
| `diary_031.jpg` | 0.9793 | 0.9795 | 0.9998 | 0.3530 |
| `diary_047.jpg` | 0.9806 | 0.9815 | 0.9991 | 0.5013 |
| `diary_050.jpg` | 0.9560 | 0.9819 | 0.9731 | 0.5750 |
| `diary_051.jpg` | 0.9782 | 0.9800 | 0.9981 | 0.4190 |
| `diary_056.jpg` | 0.9553 | 0.9911 | 0.9636 | 0.9752 |
| `diary_064.jpg` | 0.9825 | 0.9825 | 0.9999 | 0.5004 |

"""
Evaluation Script for U-Net Diary Segmentation
Evaluates the trained model strictly on the held-out test split.
Outputs:
1. Mask IoU
2. Precision
3. Recall
4. mAP@0.5 (bounding-box converted)
5. mAP@0.5:0.95
"""

import os
import argparse
import numpy as np
import torch
import cv2
import segmentation_models_pytorch as smp

def mask_to_bbox(mask):
    """Extracts (x1, y1, x2, y2) bounding box from binary mask."""
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    return np.array([x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()], dtype=float)

def compute_bbox_iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_model(weights_path, dataset_dir="dataset", img_size=(512, 384)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load model architecture
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    test_manifest = os.path.join(dataset_dir, "test_split.txt")
    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")

    with open(test_manifest, "r") as f:
        test_files = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Evaluating on {len(test_files)} frozen held-out test images...")

    mask_ious = []
    precisions = []
    recalls = []
    bbox_ious = []

    for fname in test_files:
        img_path = os.path.join(images_dir, fname)
        base = os.path.splitext(fname)[0]
        mask_path = os.path.join(masks_dir, f"{base}_mask.png")

        orig_bgr = cv2.imread(img_path)
        h_orig, w_orig = orig_bgr.shape[:2]

        img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, img_size).astype(np.float32) / 255.0
        tensor = torch.tensor(np.transpose(resized, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            pred_prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_mask = (pred_prob > 0.5).astype(np.uint8)

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask_resized = (cv2.resize(gt_mask, img_size, interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

        # Mask metrics
        intersection = np.logical_and(pred_mask, gt_mask_resized).sum()
        union = np.logical_or(pred_mask, gt_mask_resized).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        mask_ious.append(iou)

        pred_positives = pred_mask.sum()
        gt_positives = gt_mask_resized.sum()
        precision = (intersection + 1e-6) / (pred_positives + 1e-6)
        recall = (intersection + 1e-6) / (gt_positives + 1e-6)
        precisions.append(precision)
        recalls.append(recall)

        # Bounding box IoU for mAP conversion
        pred_box = mask_to_bbox(pred_mask)
        gt_box = mask_to_bbox(gt_mask_resized)
        bb_iou = compute_bbox_iou(pred_box, gt_box)
        bbox_ious.append(bb_iou)

        print(f"  {fname} | Mask IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | BBox IoU: {bb_iou:.4f}")

    mean_mask_iou = np.mean(mask_ious)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    # Compute mAP@0.5 and mAP@0.5:0.95 across bounding box IoU thresholds
    map_50 = np.mean([1.0 if b >= 0.50 else 0.0 for b in bbox_ious])
    thresholds = np.linspace(0.50, 0.95, 10)
    map_50_95 = np.mean([[1.0 if b >= t else 0.0 for t in thresholds] for b in bbox_ious])

    print("\n--- Final Test Split Evaluation Report ---")
    print(f"Mask IoU:        {mean_mask_iou:.4f}")
    print(f"Precision:       {mean_precision:.4f}")
    print(f"Recall:          {mean_recall:.4f}")
    print(f"mAP@0.5:         {map_50:.4f}")
    print(f"mAP@0.5:0.95:    {map_50_95:.4f}")

    return {
        "mask_iou": mean_mask_iou,
        "precision": mean_precision,
        "recall": mean_recall,
        "map_50": map_50,
        "map_50_95": map_50_95
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to unet_resnet34_diary.pth")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    args = parser.parse_args()

    evaluate_model(args.weights, args.dataset_dir)

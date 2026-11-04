"""
Single Command: Image to Millimetres Measurement Pipeline
Meets Requirement 4 of Measure It assessment.

Features:
1. Loads raw or new photo.
2. Undistorts lens distortion using saved calibration matrices.
3. Segments the target diary using trained U-Net model.
4. Locates the reference card (CNIC/credit card) via OpenCV contour geometry.
5. Converts pixel dimensions to millimetres using planar scale factor.
6. Cleanly handles two error states:
   - "OBJECT_NOT_FOUND"
   - "REFERENCE_NOT_FOUND"
7. Produces visual overlay with bounding box, mask, and dimension annotations.
"""

import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
import pillow_heif
import segmentation_models_pytorch as smp

pillow_heif.register_heif_opener()

# Real physical dimensions of standard reference card (ISO/IEC 7810 ID-1)
# Length: 85.0 mm, Width: 50.0 mm (verified by calliper)
REF_LENGTH_MM = 85.0
REF_WIDTH_MM = 50.0

def load_image_any(image_path):
    pil_im = Image.open(image_path)
    if pil_im.size == (4032, 3024):
        pil_im = pil_im.rotate(270, expand=True)
    img_bgr = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return img_bgr

def find_reference_card(img_bgr, diary_mask=None):
    """
    Locates the reference card (CNIC) using color thresholding and contour geometry.
    Filters out the diary area to prevent false matches.
    Returns: ((cx, cy), (dim_long, dim_short), angle), contour, or None.
    """
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Mask out the diary so its edges are never confused with the card
    if diary_mask is not None:
        dilated_diary = cv2.dilate(diary_mask, np.ones((25, 25), np.uint8))
        gray[dilated_diary > 0] = 0

    # The CNIC card is distinctly bright with high contrast against dark diary/background
    # Multi-scale adaptive thresholding
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5)
    
    # Find clean rectangular contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_card_rect = None
    best_card_contour = None
    expected_aspect = REF_LENGTH_MM / REF_WIDTH_MM  # 85.0 / 50.0 = 1.70
    
    # Target pixel area for CNIC card in 3024x4032 image typically spans 20,000 to 450,000 px
    candidate_rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15000 or area > 600000:
            continue
            
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w < 50 or h < 50:
            continue
            
        dim_long = max(w, h)
        dim_short = min(w, h)
        aspect = dim_long / dim_short
        
        # Check aspect ratio proximity to standard card aspect ratio
        aspect_diff = abs(aspect - expected_aspect)
        if aspect_diff < 0.45:
            # Check rectangular fill factor
            rect_area = dim_long * dim_short
            fill_factor = area / (rect_area + 1e-6)
            if fill_factor > 0.65:
                candidate_rects.append((aspect_diff, rect, cnt))

    if candidate_rects:
        # Pick candidate closest to true aspect ratio
        candidate_rects.sort(key=lambda x: x[0])
        _, best_card_rect, best_card_contour = candidate_rects[0]
        (cx, cy), (w, h), angle = best_card_rect
        best_card_rect = ((cx, cy), (max(w, h), min(w, h)), angle)

    return best_card_rect, best_card_contour

def measure_object(image_path, weights_path, calib_dir="calibration", output_overlay="overlay_result.jpg", apply_undistort=True):
    # Step 1: Load Image
    if not os.path.exists(image_path):
        return {"status": "ERROR", "message": f"File not found: {image_path}"}
    img = load_image_any(image_path)
    h_orig, w_orig = img.shape[:2]

    # Step 2: Undistort Image (Line where undistortion is applied for Question K2)
    if apply_undistort:
        mtx_path = os.path.join(calib_dir, "camera_matrix.npy")
        dist_path = os.path.join(calib_dir, "dist_coeffs.npy")
        if os.path.exists(mtx_path) and os.path.exists(dist_path):
            cam_mtx = np.load(mtx_path)
            dist_coeffs = np.load(dist_path)
            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(cam_mtx, dist_coeffs, (w_orig, h_orig), 1, (w_orig, h_orig))
            img = cv2.undistort(img, cam_mtx, dist_coeffs, None, newcameramtx)

    # Step 3: Segment Target Diary with U-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (512, 384)).astype(np.float32) / 255.0
    tensor = torch.tensor(np.transpose(resized, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask_small = (probs > 0.5).astype(np.uint8)

    # Scale predicted mask back to original image size
    pred_mask = cv2.resize(pred_mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    
    # Calculate confidence as mean probability over segmented foreground pixels
    fg_pixels = probs[pred_mask_small > 0]
    confidence = float(np.mean(fg_pixels)) if len(fg_pixels) > 0 else 0.0

    # Step 4: Validate Target Object Contour
    diary_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not diary_contours or cv2.contourArea(max(diary_contours, key=cv2.contourArea)) < 500000:
        return {
            "status": "OBJECT_NOT_FOUND",
            "message": "Target object (diary) was not found or segment area is too small.",
            "width_mm": None,
            "height_mm": None,
            "confidence": 0.0
        }

    dominant_diary_cnt = max(diary_contours, key=cv2.contourArea)
    diary_rect = cv2.minAreaRect(dominant_diary_cnt)
    (d_cx, d_cy), (d_w, d_h), d_angle = diary_rect
    diary_long_px = max(d_w, d_h)
    diary_short_px = min(d_w, d_h)

    # Step 5: Locate Reference Card (passing pred_mask so diary area is excluded)
    card_rect, card_cnt = find_reference_card(img, diary_mask=pred_mask)
    if card_rect is None:
        return {
            "status": "REFERENCE_NOT_FOUND",
            "message": "Reference card (CNIC) was not detected in the image.",
            "width_mm": None,
            "height_mm": None,
            "confidence": confidence
        }

    (c_cx, c_cy), (card_long_px, card_short_px), c_angle = card_rect

    # Step 6: Convert Pixels to Millimetres (Line for Question K1)
    # Valid ONLY if diary and card are co-planar and camera optical axis is orthogonal to the plane
    mm_per_pixel = REF_LENGTH_MM / card_long_px
    diary_length_mm = diary_long_px * mm_per_pixel
    diary_width_mm = diary_short_px * mm_per_pixel

    # Step 7: Create Visual Overlay
    overlay = img.copy()
    # Mask overlay in translucent green
    mask_indices = pred_mask > 0
    overlay[mask_indices] = (0.6 * overlay[mask_indices] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

    # Draw diary bounding box (cyan)
    box_diary = cv2.boxPoints(diary_rect)
    box_diary = np.int32(box_diary)
    cv2.drawContours(overlay, [box_diary], 0, (255, 255, 0), 4)

    # Draw card bounding box (yellow)
    box_card = cv2.boxPoints(card_rect)
    box_card = np.int32(box_card)
    cv2.drawContours(overlay, [box_card], 0, (0, 255, 255), 4)

    # Text annotations
    label_text = f"Diary: {diary_width_mm:.1f} x {diary_length_mm:.1f} mm | Conf: {confidence*100:.1f}%"
    cv2.putText(overlay, label_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
    cv2.putText(overlay, f"Ref Card: {REF_LENGTH_MM:.1f}x{REF_WIDTH_MM:.1f} mm ({card_long_px:.0f}px)", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

    cv2.imwrite(output_overlay, overlay)

    return {
        "status": "SUCCESS",
        "width_mm": diary_width_mm,
        "height_mm": diary_length_mm,
        "confidence": confidence,
        "scale_mm_per_px": mm_per_pixel,
        "overlay_path": output_overlay
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image to Millimetres Measurement CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input photo.")
    parser.add_argument("--weights", type=str, default="models/unet_resnet34_diary.pth", help="Path to model weights.")
    parser.add_argument("--calib_dir", type=str, default="calibration", help="Directory with camera matrix.")
    parser.add_argument("--output", type=str, default="overlay_result.jpg", help="Output path for overlay image.")
    parser.add_argument("--raw", action="store_true", help="Skip undistortion (for raw vs undistorted comparison).")
    args = parser.parse_args()

    result = measure_object(
        args.image,
        args.weights,
        args.calib_dir,
        args.output,
        apply_undistort=not args.raw
    )

    print("\n================ MEASUREMENT OUTPUT ================")
    print(f"Status:     {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"Width:      {result['width_mm']:.2f} mm")
        print(f"Height:     {result['height_mm']:.2f} mm")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Scale:      {result['scale_mm_per_px']:.5f} mm/pixel")
        print(f"Overlay:    Saved to {result['overlay_path']}")
    else:
        print(f"Reason:     {result.get('message', 'Unknown failure')}")
    print("====================================================\n")

# Camera Calibration Report

- **Camera Sensor Resolution:** 3024 x 4032 (portrait)
- **Checkerboard Specifications:** 9 x 7 inner corners, square size = 10.0 mm (measured via vernier calliper)
- **Valid Calibration Views Used:** 20 images (IMG_3086 to IMG_3120, excluding 10 landscape-oriented views)
- **OpenCV Optimization Flags:** CALIB_FIX_K2 + CALIB_FIX_K3 (constrains radial distortion to k1 to prevent edge overfitting)

## Calibration Results
- **Overall Mean RMS Reprojection Error:** 0.3641 pixels

### Camera Intrinsic Matrix (K):
```
[[3028.8291,    0.0000, 1574.8840],
 [   0.0000, 3027.2086, 1959.9940],
 [   0.0000,    0.0000,    1.0000]]
```

### Distortion Coefficients:
- k1: 0.15808552
- k2: 0.00000000 (fixed)
- p1: 0.00106638 (tangential)
- p2: 0.00430399 (tangential)
- k3: 0.00000000 (fixed)

### Verification
- Maximum corner pixel displacement on undistortion: 99.83 px (~2.5% at extreme frame boundaries).
- Straight edges preserved across frame.

# B08_v3 PT / ONNX / AM970 NPU comparison

Date: 2026-10-14. All 50 images completed on the real AM970 board.
Existing camera and static deployments were NOT replaced. No camera was started.

## Results

| Comparison | Mean mask IoU | Minimum | Matching detection counts |
| --- | ---: | ---: | ---: |
| PT vs ONNX | 100.00% | 100.00% | 50/50 |
| PT vs NPU | 99.30% | 95.54% | 47/50 |

These are output-consistency metrics, NOT accuracy against annotations.
The differing detection counts occur on 013, 018 and 042.
Small boundary and low-confidence-instance differences remain.

Against the existing annotations, mean per-image grass IoU is 98.34% for
PT/ONNX and 98.23% for NPU. For NPU alone, 018 scores 97.56% and 044
scores 95.64%. Visual inspection confirms the large foreground-road false
positive on 018 is absent, and the large left-foreground omission on 044
does not recur. This does not mean every boundary is exact.

Mean logged inference time: 36.17848 ms (36.0682 to 36.2947 ms, 50 runs).
This is the reused VKNN wrapper's inference field, not internal hardware
timing, not full postprocessing time, and not end-to-end camera FPS.
The dump tool initializes the model separately for each image.

## Method and limits

- B08_v3 best.pt, SHA256 recorded in summary.json and models/provenance.json.
- Fixed input 640x640: original width 640, height 480, top/bottom padding
  80 pixels, padding value 114. No stretching of test images.
- PT/ONNX: RGB float32 /255. NPU: BGR UINT8 through the NB preprocessing node.
- ONNX: opset 12, batch 1, no embedded NMS, fixed shape, no simplification.
- Acuity 6.33.31; target VIP9000NANODI_PLUS_PID0X1000000B.
- Existing 100 calibration images and inputmeta fitting policy reused;
  fresh B08_v3 weight calibration, not old quantization scales.
- UINT8 mixed precision with the verified seven-coordinate-node FP16 policy
  and output-concatenation FP16. Preprocessing scale is 1/255, both
  preprocessing and postprocessing nodes enabled, API outputs float32.
- All three raw outputs decoded using the same host implementation:
  confidence >=0.25, NMS IoU >0.45 suppression, mask threshold >0.5,
  corrected floor/ceil outward box crop and exclusive image bounds.
- Earlier native-PT shape evaluation used NMS 0.7 and native retina-mask
  reconstruction. Its annotation metrics are not directly interchangeable
  with this shared deployment-decoder experiment.
- PT reference uses the original project's Ultralytics 8.4.128 code;
  ONNX export uses the established conversion environment (8.3.0).
- NB, all 50 prepared inputs and all 100 board outputs passed SHA256 checks.
- Host decoding isolates backend differences; this is not a bitwise test
  of board C++ postprocessing, whose equal-score sort order may differ.
- Dataset includes 25 training and 25 validation images, not independent
  field data. No acceptance tolerance has been agreed.

## Files

- comparisons/018.jpg and 044.jpg: original / PT / ONNX / NPU, red=grass.
- vs_annotation/: annotation / PT / ONNX / NPU; green=TP, red=FP, blue=FN.
- masks/: full-size binary masks without display banners.
- summary.json and per_image.csv: backend consistency.
- accuracy_and_timing.json: annotation metrics and timing.
- reference/: confidence scores and boxes for each backend.
- models/: PT, ONNX, NB, quantization configuration and provenance.
- deployment_before.txt and deployment_after.txt: unchanged default models.

Board test directory: /root/mower_b08_v3_20261014.
Raw tensors, cached references and conversion logs remain under
/root/cpp_project/am970/model/{results,work}/ on the host.

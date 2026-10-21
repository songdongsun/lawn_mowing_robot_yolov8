# B08_v3_2 paired evaluation, 2026-10-14

Completed: native PT input-shape evaluation, ONNX export, fresh mixed-precision
calibration, real AM970 inference on 50 images, output checksum verification,
and comparison with annotations. Existing deployments remain unchanged.

## Fixed evaluation policy

- Same 25 training + 25 validation images and existing polygon annotations.
- Test input: 640x640 letterbox, original width 640 / height 480, scale 1,
  top/bottom padding 80, padding value 114. Confidence remains 0.25.
- Native PT comparison: NMS 0.7, retina_masks=True, original Ultralytics
  8.4.128 code. Actual input tensors checked. Separate 640x480 evaluation
  is included, but the reported square-input results use letterbox.
- Three-backend comparison: shared host decoder, NMS 0.45, mask >0.5,
  corrected floor/ceil outward box crop. All three use the SAME decoder.
- PT/ONNX RGB float32 /255. NPU BGR UINT8 via embedded preprocessing.
- Existing 100 calibration images and inputmeta policy reused, with fresh
  B08_v3_2 scales. UINT8 plus the verified seven coordinate-chain FP16 nodes
  and output-concatenation FP16. No extra classification precision changes.
- add_preproc_node=true, scale=1/255, add_postproc_node=true, FP32 API outputs.
- ONNX opset 12, fixed batch 1, no embedded NMS or simplification.
- Target: VIP9000NANODI_PLUS_PID0X1000000B, Acuity 6.33.31.
- Model, all 50 inputs and all 100 board outputs passed SHA256 checks.

## Native PT results against annotations

Square-input mean per-image grass IoU: B08_v3 98.05%, B08_v3_2 98.38%.
Relative to B08_v3, 40 images improve and 10 regress.
These use native PT reconstruction, not the shared deployment decoder below.

## Three-backend results

PT/ONNX final masks match on all 50 images (100% IoU).
PT/NPU mean mask IoU: 99.3081%; minimum: 96.4629% on 022.
Detection counts match on 48/50 images; 022 and 023 differ.
These are consistency metrics, not annotation accuracy.

| NPU vs annotations, per-image grass IoU | B08_v3 | B08_v3_2 |
| --- | ---: | ---: |
| All 50 mean | 98.23% | 98.33% |
| Validation 25 mean | 98.02% | 98.28% |
| 018 | 97.56% | 94.92% |
| 042 | 95.55% | 96.80% |
| 044 | 95.64% | 96.75% |

NPU-to-NPU comparison: 23 images improve, 27 regress. Total false-positive
pixels decrease from 76,596 to 62,467, while false-negative pixels increase
from 40,434 to 45,850. This is not a universal improvement.

042: the right-side supplementary instance is retained, PT confidence
0.55665 / NPU 0.65234. Its NPU box reaches x=638.625. The previous wide
missing right strip is no longer present, although boundary differences remain.
018: the large foreground-road false positive does not recur, but smaller
false-positive areas remain; the NPU annotation score regresses versus v3.
022: NPU omits a background region present in PT/ONNX. Inspect this image
and the remaining 023 instance-count mismatch before accepting the model.

Logged inference: mean 36.0124 ms, range 35.8854-36.1463 ms, 50 runs.
This wrapper timing excludes full mask postprocessing and is not camera FPS.
Each image is initialized separately by the static dump tool.

## Files and provenance

- comparisons/: original / PT / ONNX / NPU; red indicates predicted grass.
- vs_annotation/: annotation / PT / ONNX / NPU; green=TP, red=FP, blue=FN.
- vs_b08_v3_npu/: annotation / old NPU / new NPU, plus paired CSV/JSON.
- masks/: full-size lossless binary masks.
- models/: PT, ONNX, NB, quantization settings and SHA256 provenance.
- reference/: confidence scores and boxes for every image and backend.
- summary.json, per_image.csv: consistency, not ground-truth accuracy.
- accuracy_and_timing.json: annotation metrics and measured timing.
- deployment_before.txt / deployment_after.txt: default models unchanged.

Training args differ from B08_v3 in seed (0 -> 42), GPU device and run paths.
Recorded best combined box/mask epoch is 119; training logged 139 epochs.
Training args and results CSV are included under models/ for reference.

Board directory: /root/mower_b08_v3_2_20261014. No camera was started.
Raw tensors and conversion logs remain in the host am970/model project.
The dataset includes training images and is not an independent field test.
Annotations have not been independently reviewed; no acceptance tolerance
has been agreed. Equal-score NMS ordering can differ in the C++ application.

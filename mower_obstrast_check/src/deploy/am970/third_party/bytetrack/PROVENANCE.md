# ByteTrack C++ core

Source: https://github.com/FoundationVision/ByteTrack/tree/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/deploy/ncnn/cpp
Commit: d1bf0191adff59bc8fcfeaa0b33d3d1642552a99 (retrieved 2026-11-04).
Only tracker/Kalman/LAPJV files are included; NCNN detector is not used. See LICENSE.

Local adaptations: explicit thresholds; observation timestamps and millisecond expiry before association (including interrupted streams); read-only tracked/lost state access; no accumulating removed history; same-frame removal filtering; two-observation confirmation also at startup. Eight class-specific instances are owned by tracking.hpp, sharing STrack's global ID allocator. Kalman prediction remains one step per processed frame (no camera-motion compensation or variable-dt model).

LAPJV adapter: replace oversized manual allocation with RAII vectors and propagate solver errors instead of success exit/system pause.

2026-11-04 experimental person ReID: optional 256-D descriptors on Object/STrack; normalized EMA appearance history (0.9); first/high-confidence and unconfirmed association use 0.5 IoU cost + 0.5 normalized cosine distance, rejecting IoU cost >0.8 or cosine distance >0.4. Low-score second association and duplicate removal remain unchanged. Empty descriptors preserve the original path. This is a local experimental extension, not an official ByteTrack/ReID release. Features are extracted on the host and replayed into the board tracker; camera/video NPU inference does not yet supply ReID descriptors.

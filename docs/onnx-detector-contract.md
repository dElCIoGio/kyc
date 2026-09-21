# ONNX Detector Contract

`OnnxDocumentDetector` runs a box model followed by a semantic four-corner model. Both files are local, immutable deployment artifacts and must match the SHA-256 values in a JSON manifest.

## Manifest

```json
{
  "version": "ao-front-2026-01",
  "document_type": "ao_id_card",
  "side": "front",
  "box_model": {
    "path": "rtmdet-tiny.onnx",
    "sha256": "<64 hexadecimal characters>",
    "input_size": [640, 640]
  },
  "corner_model": {
    "path": "rtmpose-tiny-corners.onnx",
    "sha256": "<64 hexadecimal characters>",
    "input_size": [256, 192]
  },
  "thresholds": {
    "box_score": 0.6,
    "nms_iou": 0.4,
    "corner_score": 0.6
  }
}
```

Paths are resolved relative to the manifest. A missing file, malformed manifest, unavailable requested GPU provider, or checksum mismatch fails during detector construction.

## Tensor Contract

Both models receive one RGB float32 tensor in NCHW order with values in `[0, 1]`.

- The box input is aspect-preserving letterboxed with value 114.
- Box output has shape `N x >=5`: `x1, y1, x2, y2, score` in box-model input pixels. Runtime NMS is applied.
- Exactly one post-NMS box must remain.
- The corner input is the detected box resized to the configured corner input size.
- Corner output has shape `4 x 3`: normalized `x, y, score` in semantic TL, TR, BR, BL order.
- Every corner must meet the configured score threshold.

Training/export tooling must adapt raw RTMDet and RTMPose output to this contract. The runtime does not infer output layouts or silently reinterpret coordinates.

## Promotion Rule

The ONNX detector becomes the production default only after it beats the classical detector on the held-out private set and meets written ambiguity, corner-error, crop-validity, latency, and memory thresholds. Production must fail clearly when required artifacts are absent; it must never silently fall back.

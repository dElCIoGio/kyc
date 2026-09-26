# Optional MiniFASNet passive liveness adapter

`MiniFASNetAntiSpoofDetector` is an optional, local-only implementation of
the generic `AntiSpoofDetector` protocol. It does not collect frames, expose
HTTP endpoints, or make verification decisions. `LivenessEvaluator` remains
responsible for aggregating frame results.

Install the optional CPU PyTorch dependency explicitly:

```powershell
python -m pip install -e ".[liveness]"
```

Place the verified private runtime resources outside Git at:

```text
private-models/liveness/minifasnet/
  anti_spoof_models/
    2.7_80x80_MiniFASNetV2.pth
    4_0_0_80x80_MiniFASNetV1SE.pth
  detection_model/
    deploy.prototxt
    Widerface-RetinaFace.caffemodel
```

The detector never downloads files. It validates resources and checkpoint
compatibility when constructed, loads all models once on CPU, and has no disk
access while evaluating a frame. Inference preserves the original upstream
pipeline: RetinaFace Caffe face box, scaled crop, BGR HWC-to-CHW float tensor
in the raw 0--255 range, per-model softmax, and mean probability aggregation.
Class `1` is the real/live class; the returned score is always its mean
probability, regardless of the winning class.

The generic liveness evaluator policy remains an initial uncalibrated
development default. Model acceptance thresholds and real-capture validation
must be calibrated before production use.

The tracked adapter was adapted from Minivision's Apache-2.0
Silent-Face-Anti-Spoofing project. See [third-party notices](../THIRD_PARTY_NOTICES.md).

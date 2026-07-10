# Person Filter — Local Deployment

The person detection pre-filter runs **inline on the NAS** (no separate
service, no HTTP).  Sampled frames are scored with a local OpenCV DNN model
so the NAS can skip empty-room segments without calling the LLM.

## Backends

All backends run via OpenCV DNN, so the **only** runtime dependency is
`opencv` — no `onnxruntime` / `torch` needed.

| Backend          | Model            | Accuracy | Weight  | Notes                                   |
|------------------|------------------|----------|---------|-----------------------------------------|
| `yolov11n` (default) | `yolo11n.onnx` | High     | ~5.4 MB | YOLOv11n, fastest + most accurate on CPU. COCO person class 0. |
| `yolov26n`       | `yolo26n.onnx`   | Highest  | ~10 MB  | YOLO26n, ~30% faster than v11n on CPU, more accurate. **Optional** — see caveat below. |
| `yolov8n`        | `yolov8n.onnx`   | High     | ~12 MB  | Older but well-proven YOLOv8n.         |
| `mobilenet_ssd`  | Caffe sample     | Lower    | ~22 MB  | Older model, more false positives.     |

The YOLO ONNX export is downloaded automatically on first run to
`src/nas_video_summarizer/_person_filter_models/`.  To use a custom export
(e.g. a different YOLO variant or a quantized build), set
`PERSON_FILTER_MODEL_URL` to its download URL — it overrides the built-in
source.  The YOLO decoder auto-detects the export layout (raw `one-to-many`
head `(1, nc+4, N)` **or** end-to-end `one-to-one` head `(1, 300, 6)`), so
both work without config changes.

### `yolov26n` caveat (OpenCV version)

YOLO26 ONNX is best loaded by **OpenCV 5**.  On **OpenCV 4.x** (what
`py311-opencv` currently ships on FreeBSD) the ONNX graph may fail to load
or produce wrong output.  If the filter fails to initialize, the pipeline
**falls back to keeping all frames** (person pre-filter becomes a no-op) —
analysis still runs, only without the empty-room skip.  To use `yolov26n`
safely:

1. Check the OpenCV version on the NAS: `python -c "import cv2; print(cv2.__version__)"`.
2. If it is 5.x, set `PERSON_FILTER_BACKEND=yolov26n` and verify a run.
3. If it is 4.x, either stay on `yolov11n` (proven), upgrade to OpenCV 5,
   or install `onnxruntime` and switch the loader (breaks the zero-extra-dep
   rule — discuss first).

## Prerequisites (FreeBSD)

```bash
pkg install py311-opencv py311-numpy
```

`opencv` must be >= 4.7 for YOLOv8 ONNX support (4.8+ recommended).

For non-FreeBSD / local development you can instead pip-install:

```bash
pip install -r requirements-filter.txt
```

On first run the model file is downloaded automatically to
`src/nas_video_summarizer/_person_filter_models/`.

## Configuration (.env)

```env
PERSON_FILTER_ENABLED=true
PERSON_FILTER_BACKEND=yolov8n
PERSON_FILTER_THRESHOLD=0.3
PERSON_FILTER_SAMPLE_COUNT=12
```

- `PERSON_FILTER_SAMPLE_COUNT=12` extracts 12 frames per segment and scores
  them. After scoring, the top `SAMPLE_FRAME_COUNT` (default 6) frames are
  selected for LLM analysis.
- If **every** frame has `person_score < PERSON_FILTER_THRESHOLD`, the
  segment is skipped (no LLM call) and a `person-filter-skip` event is
  logged.
- If `opencv` is not installed or detection fails, the filter falls back to
  keeping all frames so analysis still runs.

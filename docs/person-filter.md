# Person Filter — Armbian Deployment

Deploy the MediaPipe person+face detection pre-filter on an Armbian
box.  The service receives batched JPEG frames over HTTP and returns
per-frame detection scores so that the NAS can skip empty-room segments
without calling the LLM.

## Hardware

Any ARM64 Linux board (tested on Amlogic S905D).  512 MB RAM is enough;
~100 MB extra disk for Python + models.

## Prerequisites

```bash
sudo apt update
sudo apt install -y python3.11-venv python3.11-dev
```

## Install

```bash
git clone https://github.com/suselee/mine-nas-ai-video.git /opt/mine-nas-ai-video
cd /opt/mine-nas-ai-video
python3 -m venv .venv
.venv/bin/pip install .[filter]
```

If pip can't find a pre-built mediapipe wheel for ARM64, install it
separately from a known-good source or build from source.  mediapipe
on ARM64 may require bazel and several build dependencies.

## Run (persistent via systemd)

```bash
sudo cp deploy/armbian/person-filter-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now person-filter-server
sudo systemctl status person-filter-server
```

To run manually for debugging:

```bash
.venv/bin/python3 -m nas_video_summarizer.person_filter_server \
    --host 0.0.0.0 --port 5000
```

## Configuration (environment or CLI)

| Variable / Flag            | Default | Description                     |
|----------------------------|---------|---------------------------------|
| `PERSON_FILTER_HOST`       | 0.0.0.0 | Listen address                  |
| `PERSON_FILTER_PORT`       | 5000    | Listen port                     |
| `PERSON_FILTER_OBJECT_THRESHOLD` | 0.2 | Min confidence for person (COCO class 0) |
| `PERSON_FILTER_FACE_THRESHOLD`   | 0.3 | Min confidence for face         |

## Verify

```bash
curl http://localhost:5000/health
# → {"status":"ok"}
```

## NAS side — .env

```env
PERSON_FILTER_ENABLED=true
PERSON_FILTER_URL=http://armbian-ip:5000
PERSON_FILTER_THRESHOLD=0.3
PERSON_FILTER_SAMPLE_COUNT=12
```

- `PERSON_FILTER_SAMPLE_COUNT=12` means 12 frames are extracted from each
  segment and sent to Armbian.  After scoring, the top
  `SAMPLE_FRAME_COUNT` (default 6) frames are selected for LLM analysis.

- If **every** frame has `person_score < PERSON_FILTER_THRESHOLD`, the
  segment is skipped (no LLM call) and a `person-filter-skip` event is
  logged.

## Models

| Task | Model | Speed (A53) | Size |
|------|-------|-------------|------|
| Person detection | EfficientDet-Lite0 | ~300 ms/frm | 12 MB |
| Face detection | BlazeFace-Short | ~100 ms/frm | 0.3 MB |

Change `model_selection` in `person_filter.py` if needed (0=Lite0, 1=Lite1).

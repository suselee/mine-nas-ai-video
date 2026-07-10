# Person Filter — Armbian Deployment

Deploy the MediaPipe person+face detection pre-filter on an Armbian
box.  The service receives batched JPEG frames over HTTP and returns
per-frame detection scores so that the NAS can skip empty-room segments
without calling the LLM.

## Hardware

Any ARM64 Linux board (tested on Amlogic S905D).  512 MB RAM is enough;
~100 MB extra disk for Python + models.

## Install

```bash
git clone https://github.com/suselee/mine-nas-ai-video.git /opt/mine-nas-ai-video
cd /opt/mine-nas-ai-video
python3 -m venv .venv
.venv/bin/pip install .[filter]
```

## Run (persistent via systemd)

```bash
cp deploy/armbian/person-filter-server.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now person-filter-server
systemctl status person-filter-server
```

To run manually for debugging:

```bash
person-filter-server --host 0.0.0.0 --port 5000
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

| Model by `model_selection` | Name                | Speed (A53) | Size  |
|----------------------------|---------------------|-------------|-------|
| 0 (default)                | EfficientDet-Lite0  | ~300 ms/frm | 12 MB |
| 1                          | EfficientDet-Lite1  | ~400 ms/frm | 16 MB |

Change `model_selection` in `person_filter.py` if needed.
BlazeFace-Short (face) is fixed and always runs.

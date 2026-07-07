# FreeBSD Jail Deployment

This app is meant to run inside a jail as the orchestrator. `ffmpeg` does the RTSP recording and clip cutting; Python handles state, queues, llama.cpp calls, and the web UI.

The runtime path deliberately avoids FastAPI, Pydantic, httpx, and other packages that can force Rust or native extension builds on FreeBSD. Only Python's standard library and `ffmpeg` are required to run the service.

## 1. Packages

Install Python, SQLite support, `uv`, and `ffmpeg` in the jail.

```sh
pkg install python311 py311-sqlite3 ffmpeg
```

Python 3.11 or newer is supported. If your jail already uses Python 3.12, install `python312 py312-sqlite3` instead.

Install `uv` using your preferred FreeBSD package or Python user install. Confirm it is visible to the service user:

```sh
uv --version
ffmpeg -version
ffprobe -version
```

## 2. Storage Layout

Use disk-backed storage, not memory-backed storage.

Recommended mounts:

```text
/usr/local/mine_nas_ai_video       app checkout
/var/db/nas-video                  SQLite database and rolling buffer
/mnt/nextcloud-family-moments      shared output folder visible to Nextcloud
```

Expose `/mnt/nextcloud-family-moments` to Nextcloud as external storage. Avoid writing directly into Nextcloud's internal data directory.

## 3. App Setup

```sh
cd /usr/local/mine_nas_ai_video
uv sync --no-dev
cp .env.example .env
```

Edit `.env`:

```text
DATA_DIR=/var/db/nas-video
BUFFER_DIR=/var/db/nas-video/buffer
DATABASE_PATH=/var/db/nas-video/app.sqlite3
NEXTCLOUD_OUTPUT_DIR=/mnt/nextcloud-family-moments
RTSP_USERNAME=camera-user
RTSP_PASSWORD=camera-password
RTSP_LOW_URL=rtsp://camera-ip/low-stream
RTSP_HIGH_URL=rtsp://camera-ip/4k-stream
LLAMA_BASE_URL=http://llama-jail-ip:8080/v1
LLAMA_MODEL=Qwen3-VL-2B-Instruct
```

If the RTSP password contains special characters, quote it in `.env`, for example `RTSP_PASSWORD="your#complex&password"`.

Run the preflight:

```sh
uv run nas-video-check
```

Start manually once:

```sh
uv run nas-video
```

Open `http://jail-ip:8000`.

## 4. rc.d Service

Copy the service template:

```sh
cp deploy/freebsd/nas_video /usr/local/etc/rc.d/nas_video
chmod +x /usr/local/etc/rc.d/nas_video
```

Add to `/etc/rc.conf` inside the jail:

```sh
nas_video_enable="YES"
nas_video_user="nasvideo"
nas_video_chdir="/usr/local/mine_nas_ai_video"
nas_video_uv="/usr/local/bin/uv"
nas_video_flags="run nas-video"
```

Start it:

```sh
service nas_video start
service nas_video status
```

## 5. Operating Notes

- The low-resolution RTSP stream is analyzed for speed.
- The 4K RTSP stream is kept as the high-quality source for saved clips.
- Default analysis sends one chronological contact sheet to the vision model for each 120-second segment.
- Start with the camera low stream at 640x360. If the model is still too slow, try 352x288 before 320x240.
- Raw buffer files are deleted after `RETENTION_HOURS`.
- Saved clips and JSON metadata in `NEXTCLOUD_OUTPUT_DIR` are not deleted by buffer cleanup.
- If analysis is slow, increase `RETENTION_HOURS` before lowering model quality.

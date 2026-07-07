# Agent Guide

Compact guidance for working on this repo.

## Project

FreeBSD-jail Python service that records dual RTSP streams (low-res analysis + 4K source), asks a local llama.cpp vision model whether a segment is worth keeping, and writes selected clips/metadata to a Nextcloud-visible folder.

## Tooling

- Python 3.11 or newer (see `.python-version`).
- Package manager: `uv`. Use `uv sync --no-dev` for runtime; dev group has only `pytest`.
- Tests: `uv run pytest tests/`
- No formatter/linter/typechecker is configured. Do not add one without team discussion.

## Runtime constraints

- Runtime Python dependencies are intentionally **zero** — only the standard library. Do not add FastAPI, Pydantic, httpx, or any package that requires Rust/native builds on FreeBSD.
- External binaries required on PATH: `ffmpeg`, `ffprobe`.
- Configuration lives in `.env` (copied from `.env.example`). A custom parser in `src/nas_video_summarizer/config.py` reads it; quoted values and `export ` prefixes are handled.
- Environment values already set in the shell take precedence over `.env`.

## Entrypoints

- `uv run nas-video` — start the web UI + background workers.
- `uv run nas-video-check` — preflight check of `.env`, tools, and paths.
- Both call `load_settings()` and `ensure_directories()` automatically and run DB migrations.

## Architecture

- `src/nas_video_summarizer/app.py` — `http.server` web UI and HTTP API. Workers run in a background thread with their own asyncio event loop.
- `src/nas_video_summarizer/workers.py` — `Supervisor` runs: low/high RTSP recorders, segment scanner, analyzer, and rolling-buffer cleanup.
- `src/nas_video_summarizer/ffmpeg_tools.py` — all video/audio operations (record, segment, frame extraction, contact sheets, clip concat/cut).
- `src/nas_video_summarizer/llm.py` — synchronous `urllib` client posting base64 images to a llama.cpp OpenAI-compatible `/chat/completions` endpoint; parsed from `LLAMA_BASE_URL`.
- `src/nas_video_summarizer/database.py` — SQLite with WAL and automatic migrations.
- `src/nas_video_summarizer/static/` — vanilla JS/CSS dashboard served from the package.

## Important behavior

- The app starts even if `RTSP_LOW_URL` or `RTSP_HIGH_URL` are blank; the affected recorder is disabled. Use this for local UI/DB development.
- `RTSP_USERNAME`/`RTSP_PASSWORD` are merged into the RTSP URLs inside `Settings.rtsp_low_url_for_ffmpeg` / `rtsp_high_url_for_ffmpeg` using percent-encoding. URLs that already embed credentials are left untouched.
- Default analysis sends one chronological contact sheet per segment (`ANALYSIS_IMAGE_MODE=contact_sheet`) to keep slow local models bounded. Use `frames` only for fast models.
- Saved output layout in `NEXTCLOUD_OUTPUT_DIR`: `{YYYY-MM-DD}/{HHMMSS}_{slug}.mp4`, a matching `.json` metadata file, and a `summary.md` per day.
- Raw buffer files are deleted after `RETENTION_HOURS`; saved clips and metadata are never cleaned up automatically.
- Default clips are capped at `MAX_MOMENT_SECONDS=45` with only 5s before / 10s after the detected highlight; the default prompt focuses on short indoor moments of the daughter.
- Moment videos are served with `Accept-Ranges: bytes` and `HEAD` support so browsers can seek and preload.

## Deployment

- Designed for a FreeBSD jail. See `docs/freebsd-jail.md` for packages, storage layout, and `deploy/freebsd/nas_video` rc.d template.
- The rc.d template runs `uv run nas-video` under `daemon(8)` as `nas_video_user`.

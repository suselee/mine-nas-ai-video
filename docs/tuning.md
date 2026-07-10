# Tuning — night handling, daily quota, and prompts

## Stop recording/analyzing at night (camera privacy mask)

Two independent windows, both `HH:MM` and both empty = always on. They
can cross midnight (e.g. `21:15` → `06:00`).

- `RECORD_WINDOW_START` / `RECORD_WINDOW_END` — the recorder pauses
  outside this window: RTSP is **not** pulled and no black buffer is
  written. When the window opens it resumes automatically. Use this when
  the camera masks itself at night so you stop wasting disk/CPU on
  black video.
- `ANALYSIS_WINDOW_START` / `ANALYSIS_WINDOW_END` — the analyzer skips
  entirely outside this window (no frame extraction, no LLM call).

Set both to your daytime hours (e.g. `07:00` / `21:00`). If you only set
the analysis window, the recorder still streams black video overnight but
the LLM is not called.

## Skip near-black frames any time

Regardless of the record window, every sampled frame is checked with
ffmpeg `signalstats` (average luma). If **all** frames of a segment are
near-black it is skipped (`blank-frame-skip` event) — covers a camera
that masks mid-day or a brief power-off. No extra Python dependencies.

## Save ~10–20 good clips per day

- `MOMENT_KEEP_THRESHOLD` (default `0.5`) — keep only when the model
  returns `keep=true` **and** `confidence >= threshold`. Lower = save more.
- `MAX_MOMENTS_PER_DAY` (default `0` = unlimited) — a keep-best-N daily cap.
  Once the day hits the cap, a new clip is saved only if its confidence
  beats the weakest clip already saved that day (the weakest is then
  deleted). Set `~20` to bound disk while keeping the best moments.
- Person pre-filter (`PERSON_FILTER_ENABLED=true`, default `yolov11n`) drops
  empty-room / no-person segments before they reach the LLM — this is what
  stops "empty room" clips from ever being saved.

## Prompt for a toddler (~1.5y)

The built-in `ANALYSIS_PROMPT` is tuned for a young child: it **keeps**
family interaction (with mom/grandma/dad/everyone) and the child's own
activities, and explicitly does **not** require high energy. It only
**excludes** truly low-value scenes (empty room, sleeping, passive feeding,
blank staring, screen time, blurry/black). Override `ANALYSIS_PROMPT` in
`.env` if your household differs.

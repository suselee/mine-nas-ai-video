# RV1106 MQTT integration

The RV1106 publishes daughter identity/body-track sessions to
`homecam/daughter/hit`. The NAS application subscribes with its built-in
standard-library MQTT 3.1.1 client and saves selected clips from the 4K rolling
buffer. A separate MQTT broker is required.

## Broker on the NAS

Install and configure Mosquitto on the network namespace that owns the NAS
address:

```sh
pkg install mosquitto
cp deploy/freebsd/mosquitto-nas-video.conf /usr/local/etc/mosquitto/mosquitto.conf
mosquitto_passwd -c /usr/local/etc/mosquitto/passwd nas-video
sysrc mosquitto_enable=YES
service mosquitto start
sockstat -4 -l | grep 1883
```

Use the same username and password in the RV1106 `[mqtt]` section and the NAS
`.env`. Do not expose port 1883 to the Internet.

## NAS configuration

For the recommended board-primary deployment, the NAS records only the 4K
rolling buffer. Continuous NAS analysis and the NAS low-stream recorder are
disabled; probable events still receive an event-level verification before
publication:

```env
RTSP_LOW_URL=
RTSP_HIGH_URL=rtsp://camera-ip/4k-stream
ANALYSIS_ENABLED=false
ANALYSIS_BACKEND=daughter_detector
ANALYSIS_STREAM_ROLE=high
PERSON_FILTER_ENABLED=false
RECORD_WINDOW_START=06:58
RECORD_WINDOW_END=21:02

MQTT_ENABLED=true
MQTT_HOST=192.168.123.201
MQTT_PORT=1883
MQTT_USERNAME=nas-video
MQTT_PASSWORD=replace-with-the-mosquitto-password
MQTT_DAUGHTER_TOPIC=homecam/daughter/hit
MQTT_STATUS_TOPIC=homecam/daughter/status
MQTT_CLIENT_ID=nas-video-home-camera
RV1106_SESSION_TIMEOUT_SECONDS=20
RV1106_SAVE_WAIT_SECONDS=180
RV1106_PROBABLE_POLICY=verify
```

`RV1106_PROBABLE_POLICY=verify` trusts face-confirmed sessions and samples five
frames from a staged 4K clip for probable sessions. Body-size evidence alone
cannot pass this check. Use `reject` after the board is mature enough to save
confirmed sessions only, or `accept` temporarily for maximum recall. The old
`RV1106_ACCEPT_PROBABLE` boolean remains compatible when the policy is absent.

Run `uv run nas-video-check`, restart the service, then inspect `/api/health`.
`workers.mqtt.status=connected` confirms the subscription. The RV1106 status
heartbeat appears under `workers.rv1106`.

## Delivery and recovery behavior

Fusion builds publish `start`, `update`, and `end` messages with a stable
`session_id` and sequence number. The NAS stores every accepted message in
SQLite before clip processing:

- MQTT QoS 1 duplicate deliveries are ignored by a persistent event key.
- Active and completed sessions survive NAS restarts.
- If `end` is lost, the session becomes ready after
  `RV1106_SESSION_TIMEOUT_SECONDS` without an update.
- A unique moment trigger key prevents the same session from publishing two
  clips after crash recovery.
- Legacy one-shot `event=hit` payloads remain supported and are finalized
  immediately.

The worker waits for complete 4K coverage around the board's `best_ts`; no NAS
low segment is required. Session payload, identity,
similarity/activity scores, and the persistent trigger key are written to the
moment metadata.

## Board active window

The board keeps MQTT online but closes RTSP, the decoder, and inference outside
its local active window:

```ini
[schedule]
enabled = true
start = 07:00
end = 21:00
utc_offset_minutes = 480
```

Sleeping heartbeats use `pipeline=sleeping` and `schedule_active=false`.

The retired RV1106-versus-YOLO comparison UI is no longer active. Existing
`detector_events` and `comparison_cases` tables are preserved during migration
so earlier manually reviewed results are not destroyed.

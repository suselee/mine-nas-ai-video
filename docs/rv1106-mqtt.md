# RV1106 MQTT detector comparison

The RV1106 publishes daughter face-identity hits to
`homecam/daughter/hit`. The NAS application subscribes with its built-in
standard-library MQTT 3.1.1 client. A separate broker is still required.

## Broker on the NAS

Install and configure Mosquitto on the network namespace that owns
`192.168.123.201`:

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
After the broker is running, restart `daughter_watch` once so its startup log
immediately confirms the broker connection instead of waiting for the next hit.

## Seven-day comparison configuration

Keep the existing NAS detector enabled while the board is being evaluated:

```env
ANALYSIS_BACKEND=daughter_detector
DAUGHTER_DETECTOR_MODE=heuristic

MQTT_ENABLED=true
MQTT_HOST=192.168.123.201
MQTT_PORT=1883
MQTT_USERNAME=nas-video
MQTT_PASSWORD=replace-with-the-mosquitto-password
MQTT_DAUGHTER_TOPIC=homecam/daughter/hit

DETECTOR_COMPARISON_ENABLED=true
DETECTOR_COMPARISON_DAYS=7
DETECTOR_CONTROL_SAMPLES_PER_DAY=6
```

Run `uv run nas-video-check`, restart the service, then inspect `/api/health`.
`workers.mqtt.status=connected` confirms the subscription. A real daughter hit
should create an `edge-daughter-hit` event and later a comparison case after the
corresponding high-resolution segment is stable.

During comparison, an RV1106 or NAS YOLO11n hit can save a clip. Hits within
`MQTT_EVENT_MERGE_GAP_SECONDS` are represented by one case and one video. The
dashboard labels cases as `board_only`, `yolo_only`, or `both` and provides
three review buttons. Six low-resolution negative controls are sampled from the
previous completed day and stored under `DATA_DIR/detector_comparison`; they do
not enter the Nextcloud diary or moment quotas.

After seven days, compare reviewed precision, relative union recall, and the
negative-control common miss rate. If the board is satisfactory, switch to the
low-power production mode:

```env
ANALYSIS_ENABLED=true
ANALYSIS_BACKEND=rv1106
PERSON_FILTER_ENABLED=false
MQTT_ENABLED=true
DETECTOR_COMPARISON_ENABLED=true
DETECTOR_CONTROL_SAMPLES_PER_DAY=0
```

`ANALYSIS_ENABLED` deliberately remains true: `rv1106` mode finalizes indexed
low-stream segments without running YOLO or a VLM, preventing an ever-growing
pending backlog while MQTT continues to trigger high-resolution clips.

// daughter_watch: RV1106 low-stream daughter detector.
// RockIVA provides low-cost person/face detection and stable object ids;
// RetinaFace + MobileFaceNet are scheduled only when a track needs identity.

#include <algorithm>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "config.h"
#include "facedb.h"
#include "face_detect.h"
#include "face_recog.h"
#include "h264_source.h"
#include "mpp_decoder.h"
#include "mqtt_publisher.h"
#include "rockiva_detector.h"
#include "system_monitor.h"
#include "track_fusion.h"

using namespace dw;

static volatile int g_running = 1;
static void on_signal(int) { g_running = 0; }

static double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

static int h264_nal_type(const uint8_t* data, int len) {
    int off = 0;
    if (len >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) off = 4;
    else if (len >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1) off = 3;
    return off < len ? data[off] & 0x1F : -1;
}

static std::string event_payload(const FusionEvent& event, const std::string& camera,
                                 long sequence, int width, int height,
                                 const char* pipeline) {
    int x = (int)(event.box.x1 * width);
    int y = (int)(event.box.y1 * height);
    int w = (int)((event.box.x2 - event.box.x1) * width);
    int h = (int)((event.box.y2 - event.box.y1) * height);
    char payload[1024];
    snprintf(payload, sizeof(payload),
             "{\"ts\":%.3f,\"score\":%.4f,\"camera_id\":\"%s\","
             "\"box\":[%d,%d,%d,%d],\"seq\":%ld,\"event\":\"%s\","
             "\"session_id\":\"%s\",\"session_start_ts\":%.3f,"
             "\"track_id\":%u,\"identity\":\"%s\",\"face_score\":%.4f,"
             "\"person_score\":%.4f,\"activity_score\":%.4f,"
             "\"best_ts\":%.3f,\"people_count\":%d,\"pipeline\":\"%s\"}",
             event.timestamp, event.score, camera.c_str(), x, y, w, h, sequence,
             event.event.c_str(), event.session_id.c_str(), event.session_start,
             event.track_id, event.identity.c_str(), event.face_score,
             event.person_score, event.activity_score, event.best_timestamp,
             event.people_count, pipeline);
    return payload;
}

static bool publish_legacy_face(MqttPublisher& mqtt, const std::string& topic, int qos,
                                const std::string& camera, long sequence, double now,
                                float score, const FaceBox& face, int width, int height) {
    char payload[640];
    snprintf(payload, sizeof(payload),
             "{\"ts\":%.3f,\"score\":%.4f,\"camera_id\":\"%s\","
             "\"box\":[%d,%d,%d,%d],\"seq\":%ld,\"event\":\"hit\","
             "\"identity\":\"confirmed\",\"face_score\":%.4f,"
             "\"best_ts\":%.3f,\"pipeline\":\"face_only_guard\"}",
             now, score, camera.c_str(), (int)(face.x1 * width), (int)(face.y1 * height),
             (int)((face.x2 - face.x1) * width), (int)((face.y2 - face.y1) * height),
             sequence, score, now);
    return mqtt.publish(topic, payload, qos);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <config.ini>\n", argv[0]);
        return 1;
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    Config cfg;
    if (!cfg.load(argv[1])) {
        printf("[ERR] cannot load config: %s\n", argv[1]);
        return 1;
    }

    std::string rtsp_url = cfg.get("rtsp.url");
    int rtsp_w = cfg.get_int("rtsp.width", 640);
    int rtsp_h = cfg.get_int("rtsp.height", 360);
    bool keyframes_only = cfg.get_bool("rtsp.keyframes_only", false);
    std::string pipeline_mode = cfg.get("pipeline.mode", "fusion");
    double person_fps = cfg.get_double("pipeline.person_scan_fps", 1.0);

    std::string det_path = cfg.get("model.detector");
    std::string rec_path = cfg.get("model.recognizer");
    std::string db_path = cfg.get("model.facedb");
    std::string rockiva_dir = cfg.get("model.rockiva_dir", "/root/daughter_watch/models/rockiva");

    float threshold = (float)cfg.get_double("recognize.threshold", 0.35);
    float high_threshold = (float)cfg.get_double("recognize.high_threshold", 0.55);
    int min_face = cfg.get_int("recognize.min_face", 28);
    float det_score = (float)cfg.get_double("recognize.det_score", 0.50);

    std::string mqtt_host = cfg.get("mqtt.host", "127.0.0.1");
    int mqtt_port = cfg.get_int("mqtt.port", 1883);
    std::string mqtt_user = cfg.get("mqtt.username");
    std::string mqtt_pass = cfg.get("mqtt.password");
    std::string hit_topic = cfg.get("mqtt.topic", "homecam/daughter/hit");
    std::string status_topic = cfg.get("mqtt.status_topic", "homecam/daughter/status");
    std::string mqtt_cid = cfg.get("mqtt.client_id", "rv1106");
    int mqtt_qos = cfg.get_int("mqtt.qos", 1);
    std::string camera_id = cfg.get("meta.camera_id", "home-camera");

    if (rtsp_url.empty() || det_path.empty() || rec_path.empty() || db_path.empty()) {
        printf("[ERR] rtsp.url and all face model paths are required\n");
        return 1;
    }

    FaceDetector face_detector;
    FaceRecognizer recognizer;
    FaceDB db;
    if (!face_detector.init(det_path.c_str()) || !recognizer.init(rec_path.c_str()) ||
        !db.load(db_path.c_str()) || db.empty()) {
        printf("[ERR] face detector/recognizer/database initialization failed\n");
        return 1;
    }
    printf("[INIT] facedb=%d dim=%d threshold=%.3f high=%.3f\n",
           db.count(), db.dim(), threshold, high_threshold);

    RockIvaDetector iva;
    bool fusion_enabled = pipeline_mode == "fusion" &&
        iva.init(rockiva_dir, rtsp_w, rtsp_h,
                 cfg.get_int("rockiva.person_score", 45),
                 cfg.get_int("rockiva.face_score", 45));
    if (pipeline_mode == "fusion" && !fusion_enabled)
        printf("[WARN] RockIVA unavailable (error=%d); starting face-only fallback\n", iva.last_error());

    FusionConfig fusion_cfg;
    fusion_cfg.probable_min_seconds = cfg.get_double("pipeline.probable_min_seconds", 6.0);
    fusion_cfg.probable_min_observations = cfg.get_int("pipeline.probable_min_observations", 5);
    fusion_cfg.child_max_height_ratio = cfg.get_double("pipeline.child_max_height_ratio", 0.45);
    fusion_cfg.relative_child_height_ratio = cfg.get_double("pipeline.relative_child_height_ratio", 0.75);
    fusion_cfg.face_check_interval_seconds = cfg.get_double("pipeline.face_check_interval_seconds", 2.0);
    fusion_cfg.confirmed_ttl_seconds = cfg.get_double("pipeline.confirmed_ttl_seconds", 8.0);
    fusion_cfg.track_lost_seconds = cfg.get_double("pipeline.track_lost_seconds", 6.0);
    fusion_cfg.mqtt_update_seconds = cfg.get_double("pipeline.mqtt_update_seconds", 5.0);
    fusion_cfg.face_threshold = threshold;
    fusion_cfg.face_high_threshold = high_threshold;
    TrackFusion fusion(fusion_cfg);

    MqttPublisher mqtt;
    if (!mqtt.connect(mqtt_host, mqtt_port, mqtt_cid, mqtt_user, mqtt_pass))
        printf("[WARN] MQTT initial connection failed; publish will retry\n");

    H264Source src;
    MppDecoder decoder(rtsp_w, rtsp_h);
    if (!src.open(rtsp_url) || !decoder.init()) {
        printf("[ERR] RTSP or decoder initialization failed\n");
        return 1;
    }

    SystemMonitor monitor;
    monitor.sample();
    PerformanceGuard guard(
        cfg.get_double("guard.max_cpu_percent", 65.0),
        cfg.get_int("guard.min_available_memory_mb", 80) * 1024L,
        cfg.get_double("guard.max_temperature_c", 75.0),
        cfg.get_double("guard.max_detector_p95_ms", 150.0));

    std::vector<uint8_t> chunk(256 * 1024);
    std::vector<uint8_t> rgb;
    std::vector<uint8_t> compact_nv12;
    std::vector<float> embedding;
    std::vector<double> detector_latencies;
    uint64_t pts = 0;
    uint32_t iva_frame_id = 0;
    long sequence = 0;
    long decoded_frames = 0;
    long scanned_frames = 0;
    long reconnects = 0;
    int iva_failures = 0;
    double last_scan = -1e9;
    double last_face_fallback = -1e9;
    double last_face_hit = -1e9;
    double last_health = now_seconds();
    int reconnect_wait = 2;

    printf("[RUN] %s %dx%d H264; source 5fps, person scan %.2ffps\n",
           fusion_enabled ? "rockiva_fusion_v1" : "face_only", rtsp_w, rtsp_h, person_fps);

    while (g_running) {
        int n = src.read_chunk(chunk.data(), (int)chunk.size());
        if (n < 0) {
            if (!g_running) break;
            reconnects++;
            src.close();
            decoder.deinit();
            sleep(reconnect_wait);
            if (!src.reopen() || !decoder.init()) {
                reconnect_wait = std::min(30, reconnect_wait * 2);
                continue;
            }
            reconnect_wait = 2;
            continue;
        }
        if (n > 0) {
            int nal_type = h264_nal_type(chunk.data(), n);
            bool feed = !keyframes_only || nal_type == 5 || nal_type == 6 ||
                        nal_type == 7 || nal_type == 8 || nal_type == 9;
            if (feed) decoder.send(chunk.data(), n, pts++, true);
        }

        while (true) {
            Nv12Frame frame;
            if (!decoder.get_frame(frame, 0)) break;
            decoded_frames++;
            double now = now_seconds();
            int level = guard.level();
            double effective_fps = level == 0 ? person_fps : (level == 1 ? 0.5 : 1.0);
            if (effective_fps <= 0) effective_fps = 0.5;
            if (now - last_scan < 1.0 / effective_fps) continue;
            last_scan = now;
            scanned_frames++;

            if (fusion_enabled && level < 2 && frame.yuv420_layout()) {
                IvaResult objects;
                double begin = now_seconds();
                bool ok = false;
                if (frame.data_fd() >= 0) {
                    ok = iva.detect_fd(++iva_frame_id, frame.data_fd(),
                                       frame.yuv420_layout(), frame.width(),
                                       frame.height(), objects);
                } else if (frame.physical_addr()) {
                    ok = iva.detect_physical(++iva_frame_id, frame.physical_addr(),
                                             frame.yuv420_layout(),
                                             frame.width(), frame.height(), objects);
                } else if (frame.copy_nv12(compact_nv12)) {
                    ok = iva.detect_nv12(++iva_frame_id, compact_nv12.data(), frame.width(), frame.height(), objects);
                }
                double latency = (now_seconds() - begin) * 1000.0;
                detector_latencies.push_back(latency);
                if (detector_latencies.size() > 180) detector_latencies.erase(detector_latencies.begin());
                if (!ok) {
                    printf("[ROCKIVA] detect failed error=%d\n", iva.last_error());
                    objects = IvaResult();
                    if (++iva_failures >= 3) {
                        fusion_enabled = false;
                        printf("[GUARD] RockIVA disabled after 3 consecutive failures; face-only fallback\n");
                    }
                } else {
                    iva_failures = 0;
                }
                fusion.observe(now, objects);

                uint32_t candidate_track = 0;
                for (size_t i = 0; i < objects.faces.size(); ++i) {
                    float cx = (objects.faces[i].x1 + objects.faces[i].x2) * 0.5f;
                    float cy = (objects.faces[i].y1 + objects.faces[i].y2) * 0.5f;
                    uint32_t track = fusion.track_for_face(cx, cy);
                    if (track && fusion.should_check_face(track, now)) {
                        candidate_track = track;
                        break;
                    }
                }
                if (candidate_track && frame.to_rgb(rgb)) {
                    fusion.mark_face_checked(candidate_track, now);
                    std::vector<FaceBox> faces = face_detector.detect(
                        rgb.data(), frame.width(), frame.height(), det_score);
                    for (size_t i = 0; i < faces.size(); ++i) {
                        int face_w = (int)((faces[i].x2 - faces[i].x1) * frame.width());
                        int face_h = (int)((faces[i].y2 - faces[i].y1) * frame.height());
                        if (face_w < min_face || face_h < min_face) continue;
                        if (!recognizer.extract(rgb.data(), frame.width(), frame.height(), faces[i], embedding)) continue;
                        float similarity = db.best_similarity(embedding);
                        uint32_t track = fusion.track_for_face(
                            (faces[i].x1 + faces[i].x2) * 0.5f,
                            (faces[i].y1 + faces[i].y2) * 0.5f);
                        if (track) fusion.apply_face_score(track, similarity, now);
                    }
                }

                std::vector<FusionEvent> events = fusion.collect_events(now);
                for (size_t i = 0; i < events.size(); ++i) {
                    sequence++;
                    std::string payload = event_payload(events[i], camera_id, sequence,
                                                        frame.width(), frame.height(),
                                                        "rockiva_fusion_v1");
                    bool sent = mqtt.publish(hit_topic, payload, mqtt_qos);
                    printf("[EVENT] %s track=%u identity=%s score=%.3f MQTT=%s\n",
                           events[i].event.c_str(), events[i].track_id,
                           events[i].identity.c_str(), events[i].score,
                           sent ? "OK" : "FAIL");
                }
            } else if (now - last_face_fallback >= 1.0 && frame.to_rgb(rgb)) {
                last_face_fallback = now;
                std::vector<FaceBox> faces = face_detector.detect(
                    rgb.data(), frame.width(), frame.height(), det_score);
                float best = -1;
                FaceBox best_face = {};
                for (size_t i = 0; i < faces.size(); ++i) {
                    int fw = (int)((faces[i].x2 - faces[i].x1) * frame.width());
                    int fh = (int)((faces[i].y2 - faces[i].y1) * frame.height());
                    if (fw < min_face || fh < min_face) continue;
                    if (!recognizer.extract(rgb.data(), frame.width(), frame.height(), faces[i], embedding)) continue;
                    float similarity = db.best_similarity(embedding);
                    if (similarity > best) { best = similarity; best_face = faces[i]; }
                }
                if (best >= threshold && now - last_face_hit >= 10.0) {
                    last_face_hit = now;
                    publish_legacy_face(mqtt, hit_topic, mqtt_qos, camera_id,
                                        ++sequence, now, best, best_face,
                                        frame.width(), frame.height());
                }
            }

            if (now - last_health >= 60.0) {
                last_health = now;
                SystemStats stats = monitor.sample();
                double p95 = SystemMonitor::percentile95(detector_latencies);
                int level = guard.update(stats, p95);
                char status[1024];
                snprintf(status, sizeof(status),
                         "{\"ts\":%.3f,\"camera_id\":\"%s\",\"pipeline\":\"%s\","
                         "\"guard_level\":%d,\"cpu_percent\":%.1f,"
                         "\"available_memory_mb\":%.1f,\"temperature_c\":%.1f,"
                         "\"detector_p95_ms\":%.1f,\"person_scan_fps\":%.2f,"
                         "\"active_tracks\":%d,\"confirmed_tracks\":%d,"
                         "\"probable_tracks\":%d,\"decoded_frames\":%ld,"
                         "\"scanned_frames\":%ld,\"rtsp_reconnects\":%ld}",
                         now, camera_id.c_str(), fusion_enabled ? "rockiva_fusion_v1" : "face_only",
                         level, stats.cpu_percent, stats.available_memory_kb / 1024.0,
                         stats.temperature_c, p95,
                         level == 0 ? person_fps : (level == 1 ? 0.5 : 1.0),
                         fusion.active_tracks(), fusion.confirmed_tracks(),
                         fusion.probable_tracks(), decoded_frames, scanned_frames, reconnects);
                if (!status_topic.empty()) mqtt.publish(status_topic, status, 0);
                printf("[HEALTH] cpu=%.1f%% mem=%.1fMB temp=%.1fC p95=%.1fms guard=%d\n",
                       stats.cpu_percent, stats.available_memory_kb / 1024.0,
                       stats.temperature_c, p95, level);
                detector_latencies.clear();
            }
        }
    }

    printf("[SHUTDOWN] cleaning up\n");
    src.close();
    decoder.deinit();
    mqtt.disconnect();
    iva.destroy();
    recognizer.destroy();
    face_detector.destroy();
    return 0;
}

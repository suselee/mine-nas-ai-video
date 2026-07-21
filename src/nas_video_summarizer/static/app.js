const statusGrid = document.querySelector("#status-grid");
const momentsContainer = document.querySelector("#moments");
const comparisonSummary = document.querySelector("#comparison-summary");
const comparisonCases = document.querySelector("#comparison-cases");
const refreshButton = document.querySelector("#refresh-button");
const comparisonMatchFilter = document.querySelector("#comparison-match-filter");
const comparisonReviewFilter = document.querySelector("#comparison-review-filter");
const comparisonClipFilter = document.querySelector("#comparison-clip-filter");
const comparisonOrderFilter = document.querySelector("#comparison-order-filter");
const boardReviewPreset = document.querySelector("#board-review-preset");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatDate(value) {
  if (!value) return "No data";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function renderHealth(health) {
  const latestLow = health.segments.latest_low;
  const latestHigh = health.segments.latest_high;
  const workers = health.workers;
  const analyzer = workers.analyzer || {};
  const lowRecorder = workers.recorders?.low || {};
  const highRecorder = workers.recorders?.high || {};
  const mqtt = workers.mqtt || {};
  const rv1106 = workers.rv1106 || {};
  const comparison = workers.comparison || {};

  const cards = [
    {
      label: "RTSP",
      value: health.configured.low_rtsp && health.configured.high_rtsp ? "Configured" : "Waiting",
      meta: `Low: ${health.configured.low_rtsp ? "yes" : "no"} | 4K: ${
        health.configured.high_rtsp ? "yes" : "no"
      }`,
    },
    {
      label: "Tools",
      value: health.tools.ffmpeg ? "ffmpeg ready" : "ffmpeg missing",
      meta: `ffprobe: ${health.tools.ffprobe ? "ready" : "missing"}`,
    },
    {
      label: "Recorders",
      value: `Low ${lowRecorder.status || "unknown"}`,
      meta: `4K ${highRecorder.status || "unknown"}`,
    },
    {
      label: "Analyzer",
      value: analyzer.status || "unknown",
      meta:
        analyzer.message ||
        analyzer.segment ||
        analyzer.last_segment ||
        `${health.configured.analysis_image_mode}, ${health.configured.sample_frame_count} frames`,
    },
    {
      label: "Backlog",
      value: `${health.segments.pending_low} segments`,
      meta: `Latest low: ${formatDate(latestLow?.started_at)}`,
    },
    {
      label: "Storage",
      value: `${formatBytes(health.storage.free_bytes)} free`,
      meta: `${formatBytes(health.storage.used_bytes)} used`,
    },
    {
      label: "4K Buffer",
      value: latestHigh ? "Receiving" : "No segment",
      meta: latestHigh ? formatDate(latestHigh.started_at) : health.configured.buffer_dir,
    },
    {
      label: "Output",
      value: "Nextcloud folder",
      meta: health.configured.output_dir,
    },
    {
      label: "MQTT",
      value: mqtt.status || "unknown",
      meta: mqtt.last_hit_at
        ? `Last hit: ${formatDate(mqtt.last_hit_at)}`
        : mqtt.message || `${health.configured.mqtt_host || ""}:${health.configured.mqtt_port || ""}`,
    },
    {
      label: "RV1106",
      value: rv1106.pipeline || rv1106.status || "unknown",
      meta:
        rv1106.status === "online"
          ? `CPU ${Number(rv1106.cpu_percent || 0).toFixed(1)}% · ${Number(
              rv1106.temperature_c || 0,
            ).toFixed(1)}°C · p95 ${Number(rv1106.detector_p95_ms || 0).toFixed(1)}ms · ${
              rv1106.confirmed_tracks || 0
            } confirmed / ${rv1106.probable_tracks || 0} probable · Face scans ${
              rv1106.face_scan_attempts || 0
            } · matches ${rv1106.face_track_matches || 0} · max similarity ${
              Number(rv1106.max_face_similarity) >= 0
                ? Number(rv1106.max_face_similarity).toFixed(3)
                : "—"
            }`
          : "Waiting for board heartbeat",
    },
    {
      label: "Comparison",
      value: comparison.status || "unknown",
      meta: `${comparison.metrics?.cases || 0} detector cases`,
    },
  ];

  statusGrid.innerHTML = cards
    .map(
      (card) => `
        <article class="status-card">
          <span>${escapeHtml(card.label)}</span>
          <strong>${escapeHtml(card.value)}</strong>
          <small>${escapeHtml(card.meta)}</small>
        </article>
      `,
    )
    .join("");
}

function percent(value) {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

function renderComparison(payload) {
  const metrics = payload.metrics || {};
  const board = metrics.board || {};
  const yolo = metrics.yolo || {};
  const controls = metrics.controls || {};
  comparisonSummary.innerHTML = `
    <article class="status-card"><span>RV1106</span><strong>${board.hits || 0} hits</strong><small>Precision ${percent(board.precision)} · Recall ${percent(board.relative_union_recall)} · ${metrics.identity_counts?.confirmed || 0} confirmed / ${metrics.identity_counts?.probable || 0} probable</small></article>
    <article class="status-card"><span>NAS YOLO11n</span><strong>${yolo.hits || 0} hits</strong><small>Precision ${percent(yolo.precision)} · Relative recall ${percent(yolo.relative_union_recall)}</small></article>
    <article class="status-card"><span>Agreement</span><strong>${metrics.status_counts?.both || 0} both</strong><small>${metrics.status_counts?.board_only || 0} board only · ${metrics.status_counts?.yolo_only || 0} YOLO only</small></article>
    <article class="status-card"><span>Negative controls</span><strong>${controls.reviewed || 0}/${controls.total || 0} reviewed</strong><small>Common miss rate ${percent(controls.common_miss_rate)}</small></article>
  `;

  const cases = payload.cases || [];
  if (!cases.length) {
    comparisonCases.innerHTML = `<div class="empty-state"><h3>No comparison cases yet</h3><p>Cases appear after an RV1106 or NAS detector hit.</p></div>`;
    return;
  }
  comparisonCases.innerHTML = cases
    .map((item) => {
      const source = item.control_sample ? "control" : item.match_status;
      let clipAction = `<span class="clip-pending">Clip pending</span>`;
      if (item.download_url) {
        clipAction = `<a class="button secondary" href="${item.download_url}" download>Download clip</a>`;
      } else if (item.clip_state === "skipped") {
        clipAction = `<span class="clip-pending">Skipped: ${escapeHtml(item.save_status || "not saved")}</span>`;
      }
      const faceScore = item.board_payload?.face_score;
      const pipeline = item.board_payload?.pipeline;
      const reviewControls = item.download_url
        ? `<button class="button secondary" data-review="present">Has daughter</button>
           <button class="button danger" data-review="false_positive">False positive</button>
           <button class="button secondary" data-review="uncertain">Uncertain</button>`
        : "";
      return `
        <article class="moment-card comparison-card" data-id="${item.id}">
          <div class="moment-body">
            <div class="moment-title-row"><h3>${escapeHtml(source)}</h3><span class="confidence">${escapeHtml(item.review_label)}</span></div>
            <p>${escapeHtml(formatDate(item.started_at))}</p>
            <div class="moment-meta"><span>Board ${item.board_score == null ? "—" : Number(item.board_score).toFixed(3)}</span><span>${escapeHtml(item.board_identity || "legacy")}</span><span>${escapeHtml(item.board_event_state || "hit")}</span><span>Face ${faceScore == null ? "—" : Number(faceScore).toFixed(3)}</span><span>${escapeHtml(pipeline || "legacy")}</span><span>YOLO ${item.yolo_score == null ? "—" : Number(item.yolo_score).toFixed(3)}</span></div>
            <div class="actions review-actions">
              ${clipAction}
              ${reviewControls}
            </div>
          </div>
        </article>`;
    })
    .join("");
}

function comparisonUrl() {
  const query = new URLSearchParams({ limit: "100" });
  if (comparisonMatchFilter.value) query.set("match_status", comparisonMatchFilter.value);
  if (comparisonReviewFilter.value) query.set("review_label", comparisonReviewFilter.value);
  if (comparisonClipFilter.value) query.set("clip_state", comparisonClipFilter.value);
  query.set("order", comparisonOrderFilter.value || "newest");
  return `/api/comparison?${query.toString()}`;
}

function renderMoments(moments) {
  if (!moments.length) {
    momentsContainer.innerHTML = `
      <div class="empty-state">
        <h3>No saved moments yet</h3>
        <p>After RTSP and an analysis backend are configured, selected clips will appear here automatically.</p>
      </div>
    `;
    return;
  }

  momentsContainer.innerHTML = moments
    .map((moment) => {
      const tags = (moment.tags || [])
        .map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`)
        .join("");
      return `
        <article class="moment-card" data-id="${moment.id}" data-favorited="${moment.favorited}">
          <div class="moment-body">
            <div class="moment-title-row">
              <h3>${escapeHtml(moment.title)}</h3>
              <span class="confidence">${Math.round((moment.confidence || 0) * 100)}%</span>
            </div>
            <p>${escapeHtml(moment.summary)}</p>
            <div class="tags">${tags}</div>
            <div class="moment-meta">
              <span>${escapeHtml(formatDate(moment.source_started_at))}</span>
              <span>${escapeHtml(moment.camera_name)}</span>
              <span>${escapeHtml(moment.category || moment.analysis_backend || "vlm")}</span>
            </div>
            <div class="actions">
              <a class="button secondary" href="${moment.download_url}" download>Download clip</a>
              <button class="button secondary" data-action="favorite">${
                moment.favorited ? "Unfavorite" : "Favorite"
              }</button>
              <button class="button danger" data-action="delete">Delete</button>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

async function loadAll() {
  refreshButton.disabled = true;
  try {
    const [health, moments, comparison] = await Promise.all([
      fetchJson("/api/health"),
      fetchJson("/api/moments"),
      fetchJson(comparisonUrl()),
    ]);
    renderHealth(health);
    renderMoments(moments.moments || []);
    renderComparison(comparison);
  } finally {
    refreshButton.disabled = false;
  }
}

momentsContainer.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const card = button.closest(".moment-card");
  const id = card.dataset.id;
  const action = button.dataset.action;

  if (action === "favorite") {
    const favorited = card.dataset.favorited !== "true";
    await fetchJson(`/api/moments/${id}/favorite`, {
      method: "POST",
      body: JSON.stringify({ favorited }),
    });
    await loadAll();
  }

  if (action === "delete") {
    const confirmed = window.confirm("Delete this saved clip and metadata?");
    if (!confirmed) return;
    await fetchJson(`/api/moments/${id}`, { method: "DELETE" });
    await loadAll();
  }
});

comparisonCases.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-review]");
  if (!button) return;
  const card = button.closest(".comparison-card");
  await fetchJson(`/api/comparison/${card.dataset.id}/review`, {
    method: "POST",
    body: JSON.stringify({ label: button.dataset.review }),
  });
  await loadAll();
});

refreshButton.addEventListener("click", loadAll);
for (const filter of [
  comparisonMatchFilter,
  comparisonReviewFilter,
  comparisonClipFilter,
  comparisonOrderFilter,
]) {
  filter.addEventListener("change", () => loadAll());
}
boardReviewPreset.addEventListener("click", () => {
  comparisonMatchFilter.value = "board_only";
  comparisonReviewFilter.value = "unreviewed";
  comparisonClipFilter.value = "ready";
  comparisonOrderFilter.value = "random";
  loadAll();
});
loadAll().catch((error) => {
  statusGrid.innerHTML = `<article class="status-card error"><strong>Load failed</strong><small>${escapeHtml(
    error.message,
  )}</small></article>`;
});

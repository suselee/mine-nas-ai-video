const statusGrid = document.querySelector("#status-grid");
const momentsContainer = document.querySelector("#moments");
const refreshButton = document.querySelector("#refresh-button");

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

function renderMoments(moments) {
  if (!moments.length) {
    momentsContainer.innerHTML = `
      <div class="empty-state">
        <h3>No saved moments yet</h3>
        <p>After RTSP and llama.cpp are configured, selected clips will appear here automatically.</p>
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
          <video src="${moment.video_url}" preload="metadata" controls></video>
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
            </div>
            <div class="actions">
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
    const [health, moments] = await Promise.all([
      fetchJson("/api/health"),
      fetchJson("/api/moments"),
    ]);
    renderHealth(health);
    renderMoments(moments.moments || []);
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

refreshButton.addEventListener("click", loadAll);
loadAll().catch((error) => {
  statusGrid.innerHTML = `<article class="status-card error"><strong>Load failed</strong><small>${escapeHtml(
    error.message,
  )}</small></article>`;
});

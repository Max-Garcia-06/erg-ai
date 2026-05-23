const API = "";
const STORAGE_KEY_SESSION = "tp_last_session_type";
const MAX_TABS = 10;

let sessionTypes = [];
let chartInstance = null;
let currentWorkoutId = null;
/** @type {{ id: number, title: string, cache?: object }[]} */
let workoutTabs = [];

const views = {
  history: document.getElementById("view-history"),
  upload: document.getElementById("view-upload"),
  detail: document.getElementById("view-detail"),
};

function showView(name) {
  Object.entries(views).forEach(([k, el]) => {
    if (el) el.classList.toggle("hidden", k !== name);
  });
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === name);
  });
}

async function fetchJSON(url, options = {}) {
  const res = await fetch(API + url, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function loadSessionTypes() {
  sessionTypes = await fetchJSON("/api/workouts/session-types");
  const selects = document.querySelectorAll(".session-type-select");
  const last = localStorage.getItem(STORAGE_KEY_SESSION) || "steady_state";
  selects.forEach((sel) => {
    sel.innerHTML = sessionTypes
      .map(
        (t) =>
          `<option value="${t.value}" ${t.value === last ? "selected" : ""}>${t.label}</option>`
      )
      .join("");
  });
}

function formatDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortTabDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function scoreClass(score) {
  if (score >= 90) return "grade-a";
  if (score >= 80) return "grade-b";
  if (score >= 70) return "grade-c";
  return "grade-d";
}

function tabTitle(filename, uploadedAt) {
  const base = filename.replace(/\.csv$/i, "").slice(0, 18);
  return `${base} · ${shortTabDate(uploadedAt)}`;
}

function renderTabs() {
  const container = document.getElementById("workout-tabs");
  if (!workoutTabs.length) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = workoutTabs
    .map(
      (t) => `
    <div class="tab ${t.id === currentWorkoutId ? "active" : ""}" data-id="${t.id}" role="tab">
      <button type="button" class="tab-select" title="${t.title}">${t.title}</button>
      <button type="button" class="tab-close" data-close="${t.id}" aria-label="Close tab">×</button>
    </div>`
    )
    .join("");

  container.querySelectorAll(".tab-select").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const tab = e.target.closest(".tab");
      switchToTab(parseInt(tab.dataset.id, 10));
    });
  });
  container.querySelectorAll(".tab-close").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      closeTab(parseInt(btn.dataset.close, 10));
    });
  });
}

function addTab(id, filename, uploadedAt) {
  const title = tabTitle(filename, uploadedAt);
  const existing = workoutTabs.find((t) => t.id === id);
  if (existing) {
    existing.title = title;
    return;
  }
  if (workoutTabs.length >= MAX_TABS) {
    workoutTabs.shift();
  }
  workoutTabs.push({ id, title });
}

function closeTab(id) {
  const idx = workoutTabs.findIndex((t) => t.id === id);
  if (idx === -1) return;
  workoutTabs.splice(idx, 1);
  if (currentWorkoutId === id) {
    if (workoutTabs.length) {
      const next = workoutTabs[Math.min(idx, workoutTabs.length - 1)];
      switchToTab(next.id);
    } else {
      currentWorkoutId = null;
      showView("history");
      loadHistory();
    }
  } else {
    renderTabs();
  }
}

async function switchToTab(id) {
  if (currentWorkoutId === id) return;
  currentWorkoutId = id;
  renderTabs();
  const tab = workoutTabs.find((t) => t.id === id);
  if (tab?.cache) {
    renderWorkoutDetail(tab.cache);
    return;
  }
  await loadWorkoutDetail(id);
}

async function openWorkoutTab(id, filename, uploadedAt) {
  if (filename && uploadedAt) {
    addTab(id, filename, uploadedAt);
  } else {
    const list = await fetchJSON(`/api/workouts?limit=50`);
    const item = list.find((w) => w.id === id);
    addTab(id, item?.filename || `Workout ${id}`, item?.uploaded_at || new Date().toISOString());
  }
  showView("detail");
  renderTabs();
  await switchToTab(id);
}

async function loadHistory() {
  const filter = document.getElementById("history-filter")?.value || "";
  const url = filter
    ? `/api/workouts?session_type=${filter}&limit=50`
    : "/api/workouts?limit=50";
  const items = await fetchJSON(url);
  const list = document.getElementById("history-list");
  if (!items.length) {
    list.innerHTML = `<p class="empty">No workouts yet. Upload a CSV to get started.</p>`;
    return;
  }
  list.innerHTML = items
    .map(
      (w) => `
    <article class="history-card" data-id="${w.id}">
      <div class="history-card-top">
        <span class="badge type">${w.session_label}</span>
        <span class="badge score ${scoreClass(w.overall_score || 0)}">${w.letter || "—"} ${w.overall_score ?? "—"}</span>
      </div>
      <h3>${w.filename}</h3>
      <p class="meta">${formatDate(w.uploaded_at)} · ${w.avg_power ?? "—"} W avg</p>
      <div class="chips">${(w.focus_areas || [])
        .slice(0, 2)
        .map((f) => `<span class="chip">${f.label}</span>`)
        .join("")}</div>
    </article>`
    )
    .join("");
  list.querySelectorAll(".history-card").forEach((card) => {
    card.addEventListener("click", () => {
      const id = parseInt(card.dataset.id, 10);
      const w = items.find((x) => x.id === id);
      openWorkoutTab(id, w?.filename, w?.uploaded_at);
    });
  });
}

function renderDimensionBars(rating) {
  const dims = rating.dimensions || {};
  const weights = rating.weights || {};
  const container = document.getElementById("dimension-bars");
  const entries = Object.keys(weights).filter((k) => dims[k] !== undefined);
  container.innerHTML = entries
    .map((k) => {
      const v = dims[k];
      const w = Math.round((weights[k] || 0) * 100);
      return `
      <div class="dim-row">
        <div class="dim-label"><span>${k.replace(/_/g, " ")}</span><span>${v}/100 (${w}%)</span></div>
        <div class="dim-track"><div class="dim-fill ${scoreClass(v)}" style="width:${v}%"></div></div>
      </div>`;
    })
    .join("");
}

function formatDeltaValue(metric, value) {
  if (value == null) return "—";
  if (metric === "avg_split") return formatSplit(value);
  if (metric === "avg_power") return `${value} W`;
  if (metric === "consistency" || metric === "drift") return `${value}`;
  return String(value);
}

function renderDeltaRows(containerId, deltas) {
  const el = document.getElementById(containerId);
  if (!deltas?.length) {
    el.innerHTML = `<p class="muted">No data</p>`;
    return;
  }
  el.innerHTML = deltas
    .map((d) => {
      const arrow =
        d.direction === "up" ? "↑" : d.direction === "down" ? "↓" : "→";
      const cls =
        d.direction === "na"
          ? ""
          : d.favorable
            ? "delta-good"
            : "delta-bad";
      const deltaStr =
        d.delta != null
          ? `${d.delta > 0 ? "+" : ""}${d.delta}${d.delta_pct != null ? ` (${d.delta_pct > 0 ? "+" : ""}${d.delta_pct}%)` : ""}`
          : "—";
      return `
      <div class="delta-row ${cls}">
        <span class="delta-label">${d.label}</span>
        <span class="delta-values">
          <span class="delta-current">${formatDeltaValue(d.metric, d.current)}</span>
          <span class="delta-arrow">${arrow} ${deltaStr}</span>
        </span>
      </div>`;
    })
    .join("");
}

function renderComparison(comparison, currentId) {
  const empty = document.getElementById("comparison-empty");
  const body = document.getElementById("comparison-body");

  if (!comparison?.has_prior) {
    empty.classList.remove("hidden");
    body.classList.add("hidden");
    return;
  }
  empty.classList.add("hidden");
  body.classList.remove("hidden");

  const prev = comparison.previous;
  document.getElementById("compare-prev-label").textContent = prev
    ? `${prev.filename} · ${formatDate(prev.uploaded_at)}`
    : "";

  const avg = comparison.last_5_average || {};
  document.getElementById("compare-avg-label").textContent =
    avg.count > 0
      ? `Average of ${avg.count} prior ${comparison.session_label} session${avg.count === 1 ? "" : "s"}`
      : "";

  renderDeltaRows("compare-previous", comparison.vs_previous);
  renderDeltaRows("compare-last5", comparison.vs_last_5_average);

  const tbody = document.getElementById("compare-history-body");
  const rows = comparison.prior_same_type || [];
  tbody.innerHTML = rows
    .map(
      (p) => `
    <tr>
      <td>${formatDate(p.uploaded_at)}</td>
      <td><span class="badge score ${scoreClass(p.overall_score || 0)}">${p.letter || "—"} ${p.overall_score ?? "—"}</span></td>
      <td>${p.avg_power ?? "—"} W</td>
      <td>${p.drift ?? "—"}</td>
      <td><button type="button" class="btn-link" data-open="${p.id}">Open</button></td>
    </tr>`
    )
    .join("");

  tbody.innerHTML += `
    <tr class="current-row">
      <td><strong>Current</strong></td>
      <td colspan="4" class="muted">This workout</td>
    </tr>`;

  tbody.querySelectorAll("[data-open]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const pid = parseInt(btn.dataset.open, 10);
      const p = rows.find((x) => x.id === pid);
      openWorkoutTab(pid, p?.filename, p?.uploaded_at);
    });
  });
}

function renderCoach(coach) {
  const el = document.getElementById("coach-content");
  if (!coach) {
    el.innerHTML = `<p class="muted">Click "Get coaching" for feedback.</p>`;
    return;
  }
  el.innerHTML = `
    <h3>${coach.headline}</h3>
    <div class="coach-cols">
      <div><h4>Went well</h4><ul>${(coach.went_well || []).map((x) => `<li>${x}</li>`).join("")}</ul></div>
      <div><h4>Work on</h4><ul>${(coach.work_on || []).map((x) => `<li>${x}</li>`).join("")}</ul></div>
    </div>
    <p class="next-session"><strong>Next session:</strong> ${coach.next_session}</p>
    ${coach.source ? `<p class="muted source">Source: ${coach.source}</p>` : ""}`;
}

function buildChart(series) {
  const ctx = document.getElementById("workout-chart");
  if (chartInstance) chartInstance.destroy();

  const labels = (series.time || []).map((t, i) =>
    typeof t === "number" ? t.toFixed(0) : i
  );
  const datasets = [
    {
      label: "Watts",
      data: series.watts || [],
      borderColor: "#4a9eff",
      backgroundColor: "rgba(74,158,255,0.1)",
      yAxisID: "y",
      tension: 0.2,
      pointRadius: 0,
    },
  ];
  if ((series.pace || []).some((p) => p != null)) {
    datasets.push({
      label: "Pace (s/500m)",
      data: series.pace,
      borderColor: "#ff9f43",
      yAxisID: "y1",
      tension: 0.2,
      pointRadius: 0,
    });
  }

  chartInstance = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#ccc" } } },
      scales: {
        x: { ticks: { color: "#888" }, grid: { color: "#333" } },
        y: {
          position: "left",
          title: { display: true, text: "Watts", color: "#888" },
          ticks: { color: "#888" },
          grid: { color: "#333" },
        },
        y1: {
          position: "right",
          display: datasets.length > 1,
          ticks: { color: "#888" },
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
}

function renderWorkoutDetail(w) {
  document.getElementById("detail-title").textContent = w.filename;
  document.getElementById("detail-meta").textContent = `${formatDate(w.uploaded_at)} · ${w.session_label} · ${w.detected_structure}`;

  const sel = document.getElementById("detail-session-type");
  sel.innerHTML = sessionTypes
    .map(
      (t) =>
        `<option value="${t.value}" ${t.value === w.session_type ? "selected" : ""}>${t.label}</option>`
    )
    .join("");

  const rating = w.rating;
  document.getElementById("overall-score").textContent = rating.overall ?? "—";
  document.getElementById("overall-score").className = `big-score ${scoreClass(rating.overall || 0)}`;
  document.getElementById("overall-letter").textContent = rating.letter || "";

  const warnings = document.getElementById("rating-warnings");
  const warnList = rating.warnings || [];
  const formatNote =
    rating.steady_state_format === "split_intervals" && rating.segment_count
      ? `Split steady state · ${rating.segment_count} work segments`
      : null;
  warnings.innerHTML = [
    formatNote ? `<p class="info">${formatNote}</p>` : "",
    ...warnList.map((x) => {
      const isInfo = x.toLowerCase().includes("scored as split");
      return `<p class="${isInfo ? "info" : "warning"}">${isInfo ? "ℹ" : "⚠"} ${x}</p>`;
    }),
  ].join("");

  document.getElementById("focus-chips").innerHTML = (rating.focus_areas || [])
    .map((f) => `<span class="chip warn">${f.label} (${f.score})</span>`)
    .join("");

  renderDimensionBars(rating);
  renderComparison(w.comparison, w.id);

  const s = w.summary;
  document.getElementById("stat-grid").innerHTML = `
    <div class="stat"><span>Avg power</span><strong>${s.avg_power ?? "—"} W</strong></div>
    <div class="stat"><span>Avg split</span><strong>${s.avg_split != null ? formatSplit(s.avg_split) : "—"}</strong></div>
    <div class="stat"><span>Drift</span><strong>${s.drift ?? "—"} W</strong></div>
    <div class="stat"><span>Intervals</span><strong>${s.interval_count ?? 0}</strong></div>`;

  buildChart(w.chart_series);
  renderCoach(w.coach);
}

async function loadWorkoutDetail(id) {
  const w = await fetchJSON(`/api/workouts/${id}`);
  const tab = workoutTabs.find((t) => t.id === id);
  if (tab) tab.cache = w;
  renderWorkoutDetail(w);
}

async function openDetail(id) {
  await openWorkoutTab(id);
}

function formatSplit(sec) {
  if (sec == null) return "—";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

async function analyzeUpload() {
  const fileInput = document.getElementById("file-input");
  const sessionType = document.getElementById("upload-session-type").value;
  const file = fileInput.files[0];
  if (!file) {
    alert("Choose a CSV file first.");
    return;
  }

  localStorage.setItem(STORAGE_KEY_SESSION, sessionType);
  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  btn.textContent = "Analyzing…";

  try {
    const form = new FormData();
    form.append("file", file);
    form.append("session_type", sessionType);
    const res = await fetchJSON("/api/workouts/analyze", { method: "POST", body: form });
    fileInput.value = "";
    const detail = await fetchJSON(`/api/workouts/${res.workout_id}`);
    addTab(res.workout_id, res.filename, detail.uploaded_at);
    const tab = workoutTabs.find((t) => t.id === res.workout_id);
    if (tab) tab.cache = detail;
    showView("detail");
    currentWorkoutId = res.workout_id;
    renderTabs();
    renderWorkoutDetail(detail);
    await loadHistory();
  } catch (e) {
    alert(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze workout";
  }
}

async function rescoreDetail() {
  if (!currentWorkoutId) return;
  const sessionType = document.getElementById("detail-session-type").value;
  localStorage.setItem(STORAGE_KEY_SESSION, sessionType);
  const w = await fetchJSON(`/api/workouts/${currentWorkoutId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_type: sessionType }),
  });
  const tab = workoutTabs.find((t) => t.id === currentWorkoutId);
  if (tab) {
    tab.cache = w;
    tab.title = tabTitle(w.filename, w.uploaded_at);
  }
  renderTabs();
  renderWorkoutDetail(w);
  await loadHistory();
}

async function loadCoach() {
  if (!currentWorkoutId) return;
  const btn = document.getElementById("coach-btn");
  btn.disabled = true;
  btn.textContent = "Loading…";
  try {
    const res = await fetchJSON(`/api/workouts/${currentWorkoutId}/coach`, { method: "POST" });
    renderCoach(res.coach);
    const tab = workoutTabs.find((t) => t.id === currentWorkoutId);
    if (tab?.cache) tab.cache.coach = res.coach;
  } catch (e) {
    alert(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Get coaching";
  }
}

async function deleteWorkout() {
  if (!currentWorkoutId || !confirm("Delete this workout?")) return;
  const id = currentWorkoutId;
  await fetchJSON(`/api/workouts/${id}`, { method: "DELETE" });
  closeTab(id);
  if (!workoutTabs.length) {
    showView("history");
  }
  await loadHistory();
}

function setupUploadDropzone() {
  const zone = document.getElementById("dropzone");
  const input = document.getElementById("file-input");
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      document.getElementById("file-name").textContent = e.dataTransfer.files[0].name;
    }
  });
  input.addEventListener("change", () => {
    document.getElementById("file-name").textContent = input.files[0]?.name || "No file chosen";
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const v = btn.dataset.view;
      showView(v);
      if (v === "history") await loadHistory();
    });
  });

  document.getElementById("analyze-btn")?.addEventListener("click", analyzeUpload);
  document.getElementById("rescore-btn")?.addEventListener("click", rescoreDetail);
  document.getElementById("coach-btn")?.addEventListener("click", loadCoach);
  document.getElementById("delete-btn")?.addEventListener("click", deleteWorkout);
  document.getElementById("history-filter")?.addEventListener("change", loadHistory);
  document.getElementById("back-history")?.addEventListener("click", () => {
    showView("history");
    loadHistory();
  });

  setupUploadDropzone();

  try {
    await loadSessionTypes();
    await loadHistory();
    showView("history");
  } catch (e) {
    console.error(e);
    document.getElementById("history-list").innerHTML =
      `<p class="error">Could not reach API: ${e.message}</p>`;
  }
});

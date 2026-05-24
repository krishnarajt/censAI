"""Small built-in developer UI for the CensAI pod."""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from config.settings import Config
from db.central import CentralStore

logger = logging.getLogger(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CensAI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #202322;
      --muted: #6b716f;
      --line: #d8ddd8;
      --panel: #ffffff;
      --accent: #256f63;
      --danger: #a13d3d;
      --warn: #8a641d;
      --ok: #237045;
      --info: #285b90;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    code {
      font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #39413e;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .subtitle {
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }
    main {
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 22px 24px 40px;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    button {
      min-height: 34px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 0 12px;
      font-weight: 650;
      cursor: pointer;
      white-space: nowrap;
    }
    button.secondary {
      background: white;
      color: var(--accent);
    }
    button.danger {
      border-color: var(--danger);
      background: white;
      color: var(--danger);
    }
    button:disabled {
      opacity: .55;
      cursor: wait;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 22px;
    }
    .summary {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      min-height: 74px;
    }
    .summary .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .summary .value {
      font-size: 24px;
      font-weight: 750;
      margin-top: 6px;
    }
    .section {
      margin-top: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
    }
    .section-head.wrap {
      align-items: flex-end;
      flex-wrap: wrap;
    }
    h2 {
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .control-row {
      display: grid;
      grid-template-columns: minmax(260px, 1.6fr) minmax(170px, .7fr) minmax(220px, 1fr) auto auto;
      gap: 10px;
      align-items: end;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfbf9;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    select, input {
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 9px;
      color: var(--ink);
      background: white;
      font: inherit;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #fbfbf9;
    }
    tr:last-child td { border-bottom: 0; }
    .status {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      padding: 0 9px;
      background: #edf1ee;
      color: var(--ink);
      font-size: 12px;
      font-weight: 700;
    }
    .status.detected { background: #edf1ee; color: #55605c; }
    .status.censored { background: #e8f4ed; color: var(--ok); }
    .status.failed { background: #faeded; color: var(--danger); }
    .status.rate_limited { background: #fff3d6; color: var(--warn); }
    .status.processing, .status.queued { background: #e8f0fb; color: var(--info); }
    .file {
      display: grid;
      gap: 3px;
    }
    .file strong {
      font-size: 13px;
    }
    .muted {
      color: var(--muted);
      font-size: 12px;
    }
    .row-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .error {
      color: var(--danger);
      max-height: 54px;
      overflow: auto;
    }
    .check-cell {
      text-align: center;
    }
    .check-cell input {
      width: 18px;
      min-height: 18px;
    }
    @media (max-width: 820px) {
      header { align-items: flex-start; flex-direction: column; }
      main { padding: 14px; }
      .summary-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .control-row { grid-template-columns: 1fr; }
      table { min-width: 1120px; }
      .scroll { overflow-x: auto; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>CensAI</h1>
      <div class="subtitle" id="runtimeMeta"></div>
    </div>
    <div class="toolbar">
      <button id="scanBtn" class="secondary">Scan Media</button>
      <button id="refreshBtn">Refresh</button>
    </div>
  </header>
  <main>
    <section class="summary-grid" id="summaryGrid"></section>

    <section class="section">
      <div class="section-head wrap">
        <div>
          <h2>Media Library</h2>
          <span class="meta" id="videoMeta"></span>
        </div>
        <div class="toolbar">
          <button id="queueSelectedBtn">Queue Selected</button>
          <button id="queueFolderBtn" class="secondary">Queue Folder</button>
        </div>
      </div>
      <div class="control-row">
        <label>
          Folder
          <select id="folderSelect"></select>
        </label>
        <label>
          Status
          <select id="statusFilter">
            <option value="">All</option>
            <option value="detected">Detected</option>
            <option value="queued">Queued</option>
            <option value="processing">Processing</option>
            <option value="rate_limited">Rate limited</option>
            <option value="failed">Failed</option>
            <option value="censored">Censored</option>
          </select>
        </label>
        <label>
          Search
          <input id="searchInput" placeholder="Name or path">
        </label>
        <button id="selectVisibleBtn" class="secondary">Select Visible</button>
        <button id="clearSelectionBtn" class="secondary">Clear</button>
      </div>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th style="width: 4%" class="check-cell"></th>
              <th style="width: 24%">Video</th>
              <th style="width: 10%">Status</th>
              <th style="width: 7%">Tries</th>
              <th style="width: 10%">Subtitle</th>
              <th style="width: 14%">Next Retry</th>
              <th style="width: 14%">Updated</th>
              <th style="width: 12%">Error</th>
              <th style="width: 15%"></th>
            </tr>
          </thead>
          <tbody id="videosBody"></tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>Config</h2>
        <span class="meta" id="configMeta"></span>
      </div>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th style="width: 30%">Key</th>
              <th style="width: 55%">Value</th>
              <th style="width: 15%"></th>
            </tr>
          </thead>
          <tbody id="configBody"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    let videos = [];
    let folders = [];
    let selected = new Set();
    let stats = {};

    const videosBody = document.getElementById("videosBody");
    const configBody = document.getElementById("configBody");
    const videoMeta = document.getElementById("videoMeta");
    const configMeta = document.getElementById("configMeta");
    const runtimeMeta = document.getElementById("runtimeMeta");
    const summaryGrid = document.getElementById("summaryGrid");
    const folderSelect = document.getElementById("folderSelect");
    const statusFilter = document.getElementById("statusFilter");
    const searchInput = document.getElementById("searchInput");
    const scanBtn = document.getElementById("scanBtn");
    const refreshBtn = document.getElementById("refreshBtn");
    const queueSelectedBtn = document.getElementById("queueSelectedBtn");
    const queueFolderBtn = document.getElementById("queueFolderBtn");
    const selectVisibleBtn = document.getElementById("selectVisibleBtn");
    const clearSelectionBtn = document.getElementById("clearSelectionBtn");

    function fmt(value) {
      if (!value) return "";
      const d = new Date(value);
      if (Number.isNaN(d.getTime())) return value;
      return d.toLocaleString();
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    async function api(path, options = {}) {
      const res = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }

    function statusCount(name) {
      return (stats.statuses && stats.statuses[name]) || 0;
    }

    function renderSummary() {
      const items = [
        ["Total", stats.total || 0],
        ["Detected", statusCount("detected")],
        ["Queued", statusCount("queued")],
        ["Processing", statusCount("processing")],
        ["Censored", statusCount("censored")],
        ["Rate limited", statusCount("rate_limited")],
        ["Failed", statusCount("failed")],
        ["With subtitles", stats.with_subtitles || 0],
        ["Selected", selected.size],
        ["Active", stats.queued_or_processing || 0],
      ];
      summaryGrid.innerHTML = items.map(([label, value]) => `
        <div class="summary">
          <div class="label">${escapeHtml(label)}</div>
          <div class="value">${escapeHtml(value)}</div>
        </div>
      `).join("");
    }

    function renderFolders() {
      const current = folderSelect.value;
      folderSelect.innerHTML = `<option value="">All folders</option>` + folders.map(folder => {
        const label = `${folder.label} (${folder.total})`;
        return `<option value="${escapeHtml(folder.path)}">${escapeHtml(label)}</option>`;
      }).join("");
      if ([...folderSelect.options].some(option => option.value === current)) {
        folderSelect.value = current;
      }
    }

    function filteredVideos() {
      const folder = folderSelect.value;
      const status = statusFilter.value;
      const search = searchInput.value.trim().toLowerCase();
      return videos.filter(video => {
        if (folder && !(video.path || "").startsWith(folder.endsWith("/") ? folder : folder + "/")) {
          return false;
        }
        if (status && video.status !== status) {
          return false;
        }
        if (search) {
          const haystack = `${video.name || ""} ${video.path || ""} ${video.relative_path || ""}`.toLowerCase();
          if (!haystack.includes(search)) return false;
        }
        return true;
      });
    }

    function renderVideos() {
      const rows = filteredVideos();
      videoMeta.textContent = `${rows.length} shown / ${videos.length} detected`;
      renderSummary();
      videosBody.innerHTML = "";
      for (const video of rows) {
        const checked = selected.has(video.id) ? "checked" : "";
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="check-cell"><input type="checkbox" data-id="${video.id}" ${checked}></td>
          <td>
            <div class="file">
              <strong>${escapeHtml(video.name || "")}</strong>
              <span class="muted">${escapeHtml(video.relative_path || video.path || "")}</span>
              <code>${escapeHtml(video.folder_path || "")}</code>
            </div>
          </td>
          <td><span class="status ${escapeHtml(video.status)}">${escapeHtml(video.status)}</span></td>
          <td>${video.attempts || 0}</td>
          <td>${video.has_subtitle ? "yes" : "no"}</td>
          <td>${fmt(video.next_retry_at)}</td>
          <td>${fmt(video.updated_at)}</td>
          <td class="error">${escapeHtml(video.last_error || "")}</td>
          <td>
            <div class="row-actions">
              <button class="secondary" data-action="queue" data-id="${video.id}">Queue</button>
            </div>
          </td>
        `;
        tr.querySelector("input[type=checkbox]").addEventListener("change", (event) => {
          const id = Number(event.currentTarget.dataset.id);
          if (event.currentTarget.checked) selected.add(id);
          else selected.delete(id);
          renderSummary();
        });
        tr.querySelector("button[data-action=queue]").addEventListener("click", async (event) => {
          const button = event.currentTarget;
          button.disabled = true;
          try {
            await api(`/api/videos/${button.dataset.id}/retrigger`, { method: "POST" });
            await loadVideos();
          } finally {
            button.disabled = false;
          }
        });
        videosBody.appendChild(tr);
      }
    }

    async function loadVideos() {
      const data = await api("/api/videos");
      videos = data.videos || [];
      folders = data.folders || [];
      stats = data.stats || {};
      runtimeMeta.textContent = `${data.database_backend || "db"} | ${data.llm_provider || "LLM"} | ${data.vision_model || ""}`;
      selected = new Set([...selected].filter(id => videos.some(video => video.id === id)));
      renderFolders();
      renderVideos();
    }

    async function loadConfig() {
      const data = await api("/api/config");
      configMeta.textContent = data.database_backend || "disabled";
      configBody.innerHTML = "";
      for (const item of data.configs) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${escapeHtml(item.key)}</td>
          <td><input value="${escapeHtml(item.value || "")}"></td>
          <td><button class="secondary">Save</button></td>
        `;
        tr.querySelector("button").addEventListener("click", async (event) => {
          const button = event.currentTarget;
          button.disabled = true;
          try {
            await api("/api/config", {
              method: "POST",
              body: JSON.stringify({ key: item.key, value: tr.querySelector("input").value }),
            });
            await loadConfig();
          } finally {
            button.disabled = false;
          }
        });
        configBody.appendChild(tr);
      }
    }

    async function refresh() {
      await Promise.all([loadVideos(), loadConfig()]);
    }

    scanBtn.addEventListener("click", async () => {
      scanBtn.disabled = true;
      try {
        await api("/api/scan", { method: "POST" });
        await refresh();
      } finally {
        scanBtn.disabled = false;
      }
    });
    queueSelectedBtn.addEventListener("click", async () => {
      if (!selected.size) return;
      queueSelectedBtn.disabled = true;
      try {
        await api("/api/videos/bulk-queue", {
          method: "POST",
          body: JSON.stringify({ video_ids: [...selected] }),
        });
        selected.clear();
        await loadVideos();
      } finally {
        queueSelectedBtn.disabled = false;
      }
    });
    queueFolderBtn.addEventListener("click", async () => {
      const folderPath = folderSelect.value;
      if (!folderPath) return;
      queueFolderBtn.disabled = true;
      try {
        await api("/api/folders/queue", {
          method: "POST",
          body: JSON.stringify({ folder_path: folderPath }),
        });
        await loadVideos();
      } finally {
        queueFolderBtn.disabled = false;
      }
    });
    selectVisibleBtn.addEventListener("click", () => {
      for (const video of filteredVideos()) selected.add(video.id);
      renderVideos();
    });
    clearSelectionBtn.addEventListener("click", () => {
      selected.clear();
      renderVideos();
    });
    folderSelect.addEventListener("change", renderVideos);
    statusFilter.addEventListener("change", renderVideos);
    searchInput.addEventListener("input", renderVideos);
    refreshBtn.addEventListener("click", refresh);
    refresh().catch(err => alert(err.message));
  </script>
</body>
</html>
"""


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = "CensAIUI/1.0"

    def log_message(self, fmt: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = HTML_PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _central(self) -> CentralStore:
        central = CentralStore(Config())
        central.init_db()
        return central

    def do_GET(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html()
            elif path == "/health":
                self._send_json({"ok": True})
            elif path == "/ready":
                central = self._central()
                self._send_json(
                    {
                        "ok": True,
                        "central_db_enabled": central.enabled,
                        "database_backend": central.backend_label,
                    }
                )
            elif path == "/api/videos":
                central = self._central()
                config = Config()
                self._send_json(
                    {
                        "central_db_enabled": central.enabled,
                        "database_backend": central.backend_label,
                        "llm_provider": config.llm_provider_label,
                        "vision_model": config.vision_model,
                        "profanity_model": config.profanity_model,
                        "stats": central.video_stats(),
                        "folders": central.list_folders(),
                        "videos": central.list_videos(),
                    }
                )
            elif path == "/api/config":
                central = self._central()
                self._send_json(
                    {
                        "central_db_enabled": central.enabled,
                        "database_backend": central.backend_label,
                        "configs": central.list_configs(),
                    }
                )
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            logger.exception("GET %s failed", self.path)
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            if path == "/api/scan":
                from app import scan_media_folder

                config = Config()
                result = scan_media_folder(config.MEDIA_FOLDER)
                self._send_json({"ok": True, **result})
                return

            if path == "/api/config":
                payload = self._json_body()
                central = self._central()
                central.set_config(str(payload.get("key", "")), str(payload.get("value", "")))
                Config().invalidate_runtime_config_cache()
                self._send_json({"ok": True})
                return

            if path == "/api/videos/bulk-queue":
                payload = self._json_body()
                video_ids = [int(video_id) for video_id in payload.get("video_ids", [])]
                central = self._central()
                queued = central.queue_videos(video_ids)
                self._send_json({"ok": True, "queued": queued})
                return

            if path == "/api/folders/queue":
                payload = self._json_body()
                folder_path = str(payload.get("folder_path", "")).strip()
                if not folder_path:
                    self._send_json({"error": "folder_path is required"}, HTTPStatus.BAD_REQUEST)
                    return
                central = self._central()
                queued = central.queue_folder(folder_path)
                self._send_json({"ok": True, "queued": queued})
                return

            if path.startswith("/api/videos/") and path.endswith("/retrigger"):
                video_id = int(path.split("/")[3])
                central = self._central()
                if not central.queue_video(video_id):
                    self._send_json({"error": "video not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_json({"ok": True})
                return

            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            logger.exception("POST %s failed", self.path)
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


class UIServer:
    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        config = Config()
        self.host = host or config.UI_HOST
        self.port = port or config.UI_PORT
        self._server = ThreadingHTTPServer((self.host, self.port), _RequestHandler)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("CensAI UI listening on %s:%s", self.host, self.port)

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

// DeepStream 多路视频分析运维控制台
// 数据刷新与交互逻辑

const els = {
  metricState: document.getElementById("metric-state"),
  metricStateSub: document.getElementById("metric-state-sub"),
  metricStreams: document.getElementById("metric-streams"),
  metricStreamsSub: document.getElementById("metric-streams-sub"),
  metricGpu: document.getElementById("metric-gpu"),
  metricGpuSub: document.getElementById("metric-gpu-sub"),
  metricFps: document.getElementById("metric-fps"),
  metricFpsSub: document.getElementById("metric-fps-sub"),
  metricAlerts: document.getElementById("metric-alerts"),
  metricAlertsSub: document.getElementById("metric-alerts-sub"),
  navAlertCount: document.getElementById("nav-alert-count"),
  selectedSubtitle: document.getElementById("selected-subtitle"),
  selectedStatus: document.getElementById("selected-status"),
  selectedVideo: document.getElementById("selected-video"),
  detailFps: document.getElementById("detail-fps"),
  detailLatency: document.getElementById("detail-latency"),
  detailDetections: document.getElementById("detail-detections"),
  detailDropped: document.getElementById("detail-dropped"),
  videoWallNote: document.getElementById("video-wall-note"),
  videoGrid: document.getElementById("video-grid"),
  streamList: document.getElementById("stream-list"),
  timeline: document.getElementById("timeline"),
  logs: document.getElementById("logs"),
  busState: document.getElementById("bus-state"),
  optimizationState: document.getElementById("optimization-state"),
  snapshot: document.getElementById("snapshot"),
  snapshotTime: document.getElementById("snapshot-time"),
  drawer: document.getElementById("detail-drawer"),
  drawerTitle: document.getElementById("drawer-title"),
  drawerBody: document.getElementById("drawer-body"),
  batchNote: document.getElementById("batch-note"),
  batchTotal: document.getElementById("batch-total"),
  batchOk: document.getElementById("batch-ok"),
  batchReview: document.getElementById("batch-review"),
  batchFailed: document.getElementById("batch-failed"),
  batchPersons: document.getElementById("batch-persons"),
  batchLine: document.getElementById("batch-line"),
  batchTableBody: document.getElementById("batch-table-body"),
  batchVideo: document.getElementById("batch-video"),
  batchDetail: document.getElementById("batch-detail"),
};

const state = {
  selectedStream: null,
  filter: "all",
  lastPayload: null,
  batchPayload: null,
  selectedBatchVideo: null,
  batchFilter: "all",
  localEvents: [],
  lastUpdateTime: null,
};

// 工具函数
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function statusClass(streamState) {
  if (streamState === "ERROR") return "error";
  if (streamState === "DEGRADED" || streamState === "WARNING") return "warn";
  return "ok";
}

function hasAlert(stream) {
  return Boolean(
    stream.last_error ||
    stream.last_warning ||
    stream.state === "ERROR" ||
    stream.state === "DEGRADED"
  );
}

function formatTimestamp(isoString) {
  if (!isoString) return "-";
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString("zh-CN", { hour12: false });
  } catch {
    return isoString;
  }
}

function basename(path) {
  if (!path) return "-";
  return String(path).split(/[\\/]/).filter(Boolean).pop() || String(path);
}

function qualityClass(status) {
  if (status === "failed") return "error";
  if (status === "review") return "warn";
  if (status === "passed") return "ok";
  return "info";
}

function formatRoi(value) {
  if (!value || typeof value !== "object") return "-";
  return Object.keys(value)
    .sort()
    .map((key) => `${key}=${value[key]}`)
    .join("; ");
}

function firstStream(video) {
  const streams = video?.streams || {};
  const key = Object.keys(streams).sort()[0];
  return key ? streams[key] || {} : {};
}

function batchMessages(video) {
  const quality = video?.quality || {};
  const messages = [...(quality.failures || []), ...(quality.reviews || [])];
  return messages.length ? messages.join("; ") : video?.error || "";
}

// 数据规范化和流处理
function normalizeStreams(status) {
  if (Array.isArray(status.streams) && status.streams.length > 0) {
    return status.streams;
  }

  // 如果没有 streams 数组，根据 source_count 生成 mock 数据
  const count = Number(status.source_count || 4);
  const fps = status.controllers?.fps?.last_fps || 24.0;
  return Array.from({ length: count }, (_, index) => ({
    name: `stream-${index + 1}`,
    title: `Camera ${index + 1}`,
    source: index % 2 === 0 ? `rtsp://demo/cam-${index + 1}` : `/data/sample-${index + 1}.mp4`,
    kind: index % 2 === 0 ? "rtsp" : "file",
    source_type: index % 2 === 0 ? "network" : "local",
    state: index === 2 ? "DEGRADED" : "RUNNING",
    inference_state: "active",
    playback: `rtmp://demo/live/stream-${index + 1}`,
    fps: Number((fps + index * 0.4).toFixed(1)),
    latency_ms: 44 + index * 8,
    detections: 2 + index,
    dropped_frames: index === 2 ? 7 : 0,
    last_warning: index === 2 ? "帧到达偏慢" : null,
    last_error: null,
    last_message_type: status.bus?.last_message_type || "STATE_CHANGED",
    note: "OSD 输出预览占位",
  }));
}

function selectedStream(streams) {
  if (!state.selectedStream && streams.length > 0) {
    state.selectedStream = streams[0].name;
  }
  return streams.find((item) => item.name === state.selectedStream) || streams[0] || null;
}

// 渲染系统总览指标
function renderOverview(payload, streams) {
  const status = payload.status || {};
  const monitor = status.monitor || {};
  const pipelineStatus = status.pipeline_status || {};
  const bus = status.bus || {};
  const alerts = streams.filter(hasAlert);
  const fpsValues = streams.map((item) => Number(item.fps)).filter((item) => Number.isFinite(item));
  const avgFps = fpsValues.length
    ? (fpsValues.reduce((sum, item) => sum + item, 0) / fpsValues.length).toFixed(1)
    : "-";

  const pipelineState = pipelineStatus.pipeline_state || "UNKNOWN";
  els.metricState.textContent = pipelineState;
  els.metricState.className = pipelineState === "PLAYING" ? "metric-value ok" : "metric-value";
  els.metricStateSub.textContent = `Bus: ${bus.last_message_type || "NONE"}`;

  els.metricStreams.textContent = String(status.source_count ?? streams.length);
  const runningCount = streams.filter((item) => item.state !== "ERROR").length;
  els.metricStreamsSub.textContent = `${runningCount} 路正常运行`;

  els.metricGpu.textContent = monitor.utilization_gpu == null ? "-" : `${monitor.utilization_gpu}%`;
  els.metricGpuSub.textContent = `内存 ${monitor.utilization_memory ?? "-"}% / 温度 ${monitor.temperature_c ?? "-"}°C`;

  els.metricFps.textContent = avgFps;
  els.metricFpsSub.textContent = `${status.controllers?.fps?.observations ?? 0} 次观测`;

  els.metricAlerts.textContent = String(alerts.length);
  els.metricAlerts.className = alerts.length > 0 ? "metric-value warn" : "metric-value ok";
  els.metricAlertsSub.textContent = alerts.length
    ? `${alerts[0].name}: ${alerts[0].last_warning || alerts[0].last_error || alerts[0].state}`
    : "系统运行正常";

  els.navAlertCount.textContent = String(alerts.length);
  els.busState.textContent = `Bus: ${bus.last_message_type || "NONE"}`;
  els.optimizationState.textContent = payload.optimization?.mode || "advisory_only";

  const filterLabel = state.filter === "alerts" ? "仅显示告警" : "显示全部";
  els.videoWallNote.textContent = `${streams.length} 路视频 / ${filterLabel}`;

  els.snapshotTime.textContent = formatTimestamp(payload.generated_at || status.app?.snapshot_at);
  state.lastUpdateTime = new Date();
}

// 渲染视频屏幕占位
function renderScreen(target, stream, large = false) {
  if (!stream) {
    target.innerHTML = '<div class="screen-center"><span>未选择视频流</span></div>';
    return;
  }

  const sourceLabel = `${stream.kind || "rtsp"} / ${stream.source_type || "input"}`;
  const playbackLabel = stream.playback || "输出流待接入";
  const streamTitle = stream.title || stream.name;
  const streamNote = stream.note || "OSD 结果预览";
  const sourceUri = stream.source || stream.uri || "视频源待接入";
  const stateLabel = stream.state || "RUNNING";
  const viewLabel = large ? "主视图" : "预览";

  target.innerHTML = `
    <div class="screen-overlay">
      <div class="screen-line">
        <span class="badge">${escapeHtml(sourceLabel)}</span>
        <span class="badge">${escapeHtml(playbackLabel)}</span>
      </div>
      <div class="screen-center">
        <strong>${escapeHtml(streamTitle)}</strong>
        <span>${escapeHtml(streamNote)}</span>
      </div>
      <div class="screen-line">
        <span>${escapeHtml(sourceUri)}</span>
        <span class="badge">${escapeHtml(viewLabel)} / ${escapeHtml(stateLabel)}</span>
      </div>
    </div>
  `;
}

// 渲染当前选中的视频详情
function renderSelected(stream) {
  if (!stream) return;

  els.selectedSubtitle.textContent = `${stream.name} / ${stream.source || "视频源待接入"}`;
  els.selectedStatus.textContent = stream.state || "UNKNOWN";
  els.selectedStatus.className = `status ${statusClass(stream.state)}`;

  els.detailFps.textContent = stream.fps ?? "-";
  els.detailLatency.textContent = stream.latency_ms == null ? "-" : `${stream.latency_ms} ms`;
  els.detailDetections.textContent = stream.detections ?? 0;
  els.detailDropped.textContent = stream.dropped_frames ?? 0;

  renderScreen(els.selectedVideo, stream, true);
}

// 渲染视频墙
function renderVideoWall(streams) {
  const visibleStreams = state.filter === "alerts" ? streams.filter(hasAlert) : streams;

  if (visibleStreams.length === 0) {
    els.videoGrid.innerHTML = '<div style="grid-column: 1/-1; padding: 40px; text-align: center; color: var(--muted);">当前筛选条件下无视频流</div>';
    return;
  }

  els.videoGrid.innerHTML = visibleStreams
    .map((stream) => {
      const selected = stream.name === state.selectedStream;
      const statusCls = statusClass(stream.state);
      return `
      <article class="video-tile ${selected ? "selected" : ""}" data-stream="${escapeHtml(stream.name)}">
        <div class="video-screen" data-screen="${escapeHtml(stream.name)}"></div>
        <div class="tile-meta">
          <strong>${escapeHtml(stream.title || stream.name)}</strong>
          <span class="status ${statusCls}">${escapeHtml(stream.state || "RUNNING")}</span>
        </div>
        <div class="tile-meta">
          <span>FPS: ${stream.fps ?? "-"} / 延迟: ${stream.latency_ms ?? "-"} ms</span>
          <span>检测: ${stream.detections ?? 0} / 丢帧: ${stream.dropped_frames ?? 0}</span>
        </div>
      </article>
    `;
    })
    .join("");

  // 为每个视频屏幕渲染占位内容
  visibleStreams.forEach((stream) => {
    const screen = document.querySelector(`[data-screen="${CSS.escape(stream.name)}"]`);
    if (screen) renderScreen(screen, stream);
  });
}

// 渲染多路任务监控列表
function renderStreamList(streams) {
  const visibleStreams = state.filter === "alerts" ? streams.filter(hasAlert) : streams;

  if (visibleStreams.length === 0) {
    els.streamList.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--muted);">当前筛选条件下无任务</div>';
    return;
  }

  els.streamList.innerHTML = visibleStreams
    .map((stream) => {
      const selected = stream.name === state.selectedStream;
      const statusCls = statusClass(stream.state);
      const alertText = stream.last_warning || stream.last_error || stream.state || "OK";

      return `
    <div class="stream-row ${selected ? "selected" : ""}" data-stream="${escapeHtml(stream.name)}">
      <div>
        <strong>${escapeHtml(stream.title || stream.name)}</strong>
        <p>
          ${escapeHtml(stream.source || "视频源待接入")}<br>
          ${escapeHtml(stream.kind || "rtsp")} / 推理: ${escapeHtml(stream.inference_state || "active")} / Bus: ${escapeHtml(stream.last_message_type || "none")}
        </p>
      </div>
      <div class="row-kpi">
        <strong>${stream.fps ?? "-"}</strong>
        <span class="status ${statusCls}">${escapeHtml(alertText)}</span>
      </div>
    </div>
  `;
    })
    .join("");
}

// 渲染事件时间线和日志
function renderEvents(payload, streams) {
  const status = payload.status || {};
  const bus = status.bus || {};
  const pipelineStatus = status.pipeline_status || {};

  const operationEvents = [
    {
      time: "PIPELINE",
      text: `Pipeline ${pipelineStatus.pipeline_state || "UNKNOWN"} / 运行中: ${status.is_running ? "是" : "否"}`,
    },
    {
      time: "BUS",
      text: `${bus.last_message_type || "NONE"} ${bus.last_warning || bus.last_error || ""}`.trim(),
    },
    {
      time: "PROBE",
      text: "Probe 回调链路已预留，等待真实 NvDsBatchMeta 数据接入",
    },
    ...streams
      .filter(hasAlert)
      .slice(0, 3)
      .map((stream) => ({
        time: stream.name,
        text: `${stream.title || stream.name}: ${stream.last_warning || stream.last_error || stream.state}`,
      })),
    ...state.localEvents.slice(-4),
  ];

  els.timeline.innerHTML = operationEvents
    .map(
      (event) => `
    <div class="event">
      <time>${escapeHtml(event.time)}</time>
      <span>${escapeHtml(event.text)}</span>
    </div>
  `
    )
    .join("");

  const logs = payload.recent_logs?.items || [];
  if (logs.length === 0) {
    els.logs.textContent = "暂无日志数据";
  } else {
    els.logs.textContent = logs
      .map((item) => `[${item.timestamp || "-"}] ${item.level || "INFO"} ${item.message || ""}`)
      .join("\n");
  }
}

// 渲染单路详情抽屉
function renderDrawer(stream) {
  if (!stream) return;

  els.drawerTitle.textContent = `${stream.title || stream.name} - 详细信息`;

  const statusCls = statusClass(stream.state);
  const inferenceStatus = stream.inference_state === "active" ? "运行中" : stream.inference_state || "未知";
  const hasWarnings = stream.last_warning || stream.last_error;

  els.drawerBody.innerHTML = `
    <section class="drawer-section">
      <h3>视频源配置</h3>
      <p>
        <strong>通道名称:</strong> ${escapeHtml(stream.name)}<br>
        <strong>显示名称:</strong> ${escapeHtml(stream.title || stream.name)}<br>
        <strong>输入源:</strong> ${escapeHtml(stream.source || "视频源待接入")}<br>
        <strong>输出流:</strong> ${escapeHtml(stream.playback || "输出流待接入")}<br>
        <strong>源类型:</strong> ${escapeHtml(stream.kind || "rtsp")} / ${escapeHtml(stream.source_type || "unknown")}
      </p>
    </section>
    <section class="drawer-section">
      <h3>运行状态</h3>
      <p>
        <strong>Pipeline 状态:</strong> <span class="status ${statusCls}">${escapeHtml(stream.state || "RUNNING")}</span><br>
        <strong>推理状态:</strong> ${escapeHtml(inferenceStatus)}<br>
        <strong>Bus 消息:</strong> ${escapeHtml(stream.last_message_type || "none")}<br>
        <strong>最后更新:</strong> ${formatTimestamp(stream.last_update || new Date().toISOString())}
      </p>
    </section>
    <section class="drawer-section">
      <h3>性能指标</h3>
      <p>
        <strong>当前 FPS:</strong> ${stream.fps ?? "-"}<br>
        <strong>端到端延迟:</strong> ${stream.latency_ms ?? "-"} ms<br>
        <strong>检测对象数:</strong> ${stream.detections ?? 0}<br>
        <strong>累计丢帧数:</strong> ${stream.dropped_frames ?? 0}<br>
        <strong>丢帧率:</strong> ${stream.dropped_frames && stream.fps ? ((stream.dropped_frames / (stream.fps * 10)) * 100).toFixed(2) : "0.00"}%
      </p>
    </section>
    <section class="drawer-section">
      <h3>告警与异常</h3>
      <p>
        <strong>最近警告:</strong> ${hasWarnings ? '<span class="status warn">' + escapeHtml(stream.last_warning || "无") + '</span>' : "无"}<br>
        <strong>最近错误:</strong> ${stream.last_error ? '<span class="status error">' + escapeHtml(stream.last_error) + '</span>' : "无"}<br>
        <strong>告警状态:</strong> ${hasWarnings ? '<span class="status warn">需要关注</span>' : '<span class="status ok">正常</span>'}<br>
        <strong>备注:</strong> ${escapeHtml(stream.note || "当前无活动告警")}
      </p>
    </section>
    <section class="drawer-section">
      <h3>快速操作</h3>
      <p style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
        <button class="btn" type="button" data-action="restart-stream" data-stream="${escapeHtml(stream.name)}">重启此路</button>
        <button class="btn" type="button" data-action="pause-stream" data-stream="${escapeHtml(stream.name)}">暂停推理</button>
        <button class="btn danger" type="button" data-action="stop-stream" data-stream="${escapeHtml(stream.name)}">停止此路</button>
      </p>
    </section>
  `;
}

// 渲染调试快照
function renderSnapshot(payload) {
  els.snapshot.textContent = JSON.stringify(payload, null, 2);
}

function renderBatchDashboard(payload) {
  state.batchPayload = payload;
  const summary = payload.summary || {};
  const quality = payload.quality || {};
  const videos = payload.videos || [];

  els.batchNote.textContent = payload.batch_dir
    ? `批量目录: ${payload.batch_dir}`
    : "尚未生成批量结果";
  els.batchTotal.textContent = String(summary.video_count ?? videos.length ?? 0);
  els.batchOk.textContent = String(summary.processed_count ?? 0);
  els.batchReview.textContent = String(quality.review_count ?? 0);
  els.batchFailed.textContent = String(quality.failed_count ?? summary.failed_count ?? 0);
  els.batchPersons.textContent = String(summary.total_unique_persons_sum ?? 0);
  els.batchLine.textContent = `${summary.line_crossing_in_sum ?? 0} / ${summary.line_crossing_out_sum ?? 0}`;

  if (!state.selectedBatchVideo && videos.length > 0) {
    state.selectedBatchVideo = String(videos[0].index || 1);
  }

  renderBatchTable(videos);
  renderBatchDetail(videos.find((video) => String(video.index) === state.selectedBatchVideo) || videos[0]);
}

function renderBatchTable(videos) {
  const visible = videos.filter((video) => {
    const status = video.quality?.quality_status || "unknown";
    if (state.batchFilter === "all") return true;
    return status === state.batchFilter;
  });

  if (!visible.length) {
    els.batchTableBody.innerHTML = '<tr><td colspan="10">当前筛选条件下没有批量结果</td></tr>';
    return;
  }

  els.batchTableBody.innerHTML = visible
    .map((video) => {
      const stream = firstStream(video);
      const qualityStatus = video.quality?.quality_status || "unknown";
      const selected = String(video.index) === state.selectedBatchVideo;
      return `
        <tr class="${selected ? "selected" : ""}" data-batch-video="${escapeHtml(video.index)}">
          <td>${escapeHtml(video.index)}</td>
          <td title="${escapeHtml(video.input_video)}">${escapeHtml(basename(video.input_video))}</td>
          <td><span class="status ${video.status === "ok" ? "ok" : "error"}">${escapeHtml(video.status || "unknown")}</span></td>
          <td><span class="status ${qualityClass(qualityStatus)}">${escapeHtml(qualityStatus)}</span></td>
          <td>${escapeHtml(video.total_unique_persons ?? 0)}</td>
          <td>${escapeHtml(formatRoi(video.roi_unique_persons))}</td>
          <td>${escapeHtml(video.line_crossing_in ?? 0)} / ${escapeHtml(video.line_crossing_out ?? 0)}</td>
          <td>${escapeHtml(stream.estimated_fps ?? "-")}</td>
          <td>${escapeHtml(stream.is_frame_continuous ?? "-")}</td>
          <td title="${escapeHtml(batchMessages(video))}">${escapeHtml(batchMessages(video) || "-")}</td>
        </tr>
      `;
    })
    .join("");
}

function renderBatchDetail(video) {
  if (!video) {
    els.batchVideo.removeAttribute("src");
    els.batchDetail.innerHTML = '<div class="empty-state">未找到批量结果。先运行 scripts/run_person_analytics_batch.sh。</div>';
    return;
  }

  const stream = firstStream(video);
  const qualityStatus = video.quality?.quality_status || "unknown";
  if (video.output_overlay_video_url) {
    if (els.batchVideo.getAttribute("src") !== video.output_overlay_video_url) {
      els.batchVideo.setAttribute("src", video.output_overlay_video_url);
      els.batchVideo.load();
    }
  } else {
    els.batchVideo.removeAttribute("src");
  }

  const links = [
    ["video", video.output_video_url],
    ["overlay", video.output_overlay_video_url],
    ["jsonl", video.output_jsonl_url],
    ["summary", video.output_summary_url],
  ]
    .filter(([, url]) => Boolean(url))
    .map(([label, url]) => `<a class="btn" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`)
    .join("");

  els.batchDetail.innerHTML = `
    <div class="batch-detail-row"><span>视频</span><strong>${escapeHtml(basename(video.input_video))}</strong></div>
    <div class="batch-detail-row"><span>运行状态</span><strong>${escapeHtml(video.status || "unknown")}</strong></div>
    <div class="batch-detail-row"><span>质量状态</span><strong>${escapeHtml(qualityStatus)}</strong></div>
    <div class="batch-detail-row"><span>去重人数</span><strong>${escapeHtml(video.total_unique_persons ?? 0)}</strong></div>
    <div class="batch-detail-row"><span>ROI 人数</span><strong>${escapeHtml(formatRoi(video.roi_unique_persons))}</strong></div>
    <div class="batch-detail-row"><span>越线 In/Out</span><strong>${escapeHtml(video.line_crossing_in ?? 0)} / ${escapeHtml(video.line_crossing_out ?? 0)}</strong></div>
    <div class="batch-detail-row"><span>FPS</span><strong>${escapeHtml(stream.estimated_fps ?? "-")}</strong></div>
    <div class="batch-detail-row"><span>帧连续</span><strong>${escapeHtml(stream.is_frame_continuous ?? "-")}</strong></div>
    <div class="batch-detail-row"><span>原因</span><strong>${escapeHtml(batchMessages(video) || "无")}</strong></div>
    <div class="batch-links">${links || '<span class="empty-state">暂无输出链接</span>'}</div>
  `;
}

// 添加本地事件
function addLocalEvent(text) {
  const timestamp = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  state.localEvents.push({ time: "UI", text: `[${timestamp}] ${text}` });
  if (state.localEvents.length > 12) {
    state.localEvents.shift();
  }
  if (state.lastPayload) render(state.lastPayload);
}

// 主渲染函数
function render(payload) {
  state.lastPayload = payload;
  const status = payload.status || {};
  const streams = normalizeStreams(status);
  const current = selectedStream(streams);

  renderOverview(payload, streams);
  renderSelected(current);
  renderVideoWall(streams);
  renderStreamList(streams);
  renderEvents(payload, streams);
  renderDrawer(current);
  renderSnapshot(payload);
}

// 从 API 刷新数据
async function refresh() {
  try {
    const response = await fetch("/api/debug?limit=80");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const payload = await response.json();
    render(payload);
  } catch (error) {
    console.error("加载控制台数据失败:", error);
    els.snapshot.textContent = `加载控制台数据失败:\n${error.message}\n\n请检查后端服务是否正常运行。`;
  }
}

async function refreshBatchDashboard() {
  try {
    const response = await fetch("/api/batch/dashboard");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const payload = await response.json();
    renderBatchDashboard(payload);
  } catch (error) {
    console.warn("加载批量结果失败:", error);
    els.batchTableBody.innerHTML = `<tr><td colspan="10">批量结果未就绪：${escapeHtml(error.message)}</td></tr>`;
    els.batchDetail.innerHTML = '<div class="empty-state">先运行批量处理，生成 outputs/batch/batch_summary.json 和 batch_quality.json。</div>';
  }
}

// 事件监听 - 视频流选择和筛选
document.addEventListener("click", (event) => {
  // 选择视频流
  const streamTarget = event.target.closest("[data-stream]");
  if (streamTarget) {
    state.selectedStream = streamTarget.dataset.stream;
    if (state.lastPayload) render(state.lastPayload);
    return;
  }

  // 筛选按钮
  const filterButton = event.target.closest("[data-filter]");
  if (filterButton) {
    state.filter = filterButton.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((button) => button.classList.remove("active"));
    filterButton.classList.add("active");
    if (state.lastPayload) render(state.lastPayload);
    return;
  }

  const batchRow = event.target.closest("[data-batch-video]");
  if (batchRow) {
    state.selectedBatchVideo = batchRow.dataset.batchVideo;
    if (state.batchPayload) renderBatchDashboard(state.batchPayload);
    return;
  }

  const batchFilterButton = event.target.closest("[data-batch-filter]");
  if (batchFilterButton) {
    state.batchFilter = batchFilterButton.dataset.batchFilter;
    document.querySelectorAll("[data-batch-filter]").forEach((button) => button.classList.remove("active"));
    batchFilterButton.classList.add("active");
    if (state.batchPayload) renderBatchDashboard(state.batchPayload);
    return;
  }

  // 操作按钮
  const actionButton = event.target.closest("[data-action]");
  if (actionButton) {
    const action = actionButton.dataset.action;
    const actionLabel = actionButton.textContent.trim();
    addLocalEvent(`${actionLabel} 操作已记录 (当前为 Mock/Demo 预览模式，未连接真实 Pipeline)`);
    return;
  }
});

// 打开详情抽屉
document.getElementById("open-detail").addEventListener("click", () => {
  els.drawer.classList.add("open");
  els.drawer.setAttribute("aria-hidden", "false");
});

// 关闭详情抽屉
document.getElementById("close-detail").addEventListener("click", () => {
  els.drawer.classList.remove("open");
  els.drawer.setAttribute("aria-hidden", "true");
});

// 确认告警
document.getElementById("mark-alert-read").addEventListener("click", () => {
  addLocalEvent(`已确认 ${state.selectedStream || "当前选中视频"} 的告警状态`);
});

// 初始化：立即刷新一次，然后每秒自动刷新
refresh();
refreshBatchDashboard();
window.setInterval(refresh, 1000);
window.setInterval(refreshBatchDashboard, 5000);

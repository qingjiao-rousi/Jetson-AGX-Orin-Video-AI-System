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
  batchDuration: document.getElementById("batch-duration"),
  batchJobs: document.getElementById("batch-jobs"),
  batchProcessingFps: document.getElementById("batch-processing-fps"),
  batchArtifacts: document.getElementById("batch-artifacts"),
  acceptanceTitle: document.getElementById("acceptance-title"),
  acceptanceDetail: document.getElementById("acceptance-detail"),
  batchTableBody: document.getElementById("batch-table-body"),
  batchVideo: document.getElementById("batch-video"),
  batchDetail: document.getElementById("batch-detail"),
  multifileNote: document.getElementById("multifile-note"),
  multifileArtifacts: document.getElementById("multifile-artifacts"),
  multifileStatus: document.getElementById("multifile-status"),
  multifileStreams: document.getElementById("multifile-streams"),
  multifileFrames: document.getElementById("multifile-frames"),
  multifileDetections: document.getElementById("multifile-detections"),
  multifilePersons: document.getElementById("multifile-persons"),
  multifileReview: document.getElementById("multifile-review"),
  multifileFailed: document.getElementById("multifile-failed"),
  multifileTableBody: document.getElementById("multifile-table-body"),
  multifileIndividualVideos: document.getElementById("multifile-individual-videos"),
  multifileDetail: document.getElementById("multifile-detail"),
};

if (els.batchVideo) {
  els.batchVideo.addEventListener("error", () => {
    const current = els.batchVideo.currentSrc || els.batchVideo.getAttribute("src") || "";
    const notice = document.createElement("div");
    notice.className = "empty-state";
    notice.textContent = `当前浏览器无法播放此视频编码：${current}`;
    const existing = els.batchDetail.querySelector("[data-video-error]");
    if (existing) existing.remove();
    notice.dataset.videoError = "1";
    els.batchDetail.prepend(notice);
  });
}

const state = {
  selectedStream: null,
  filter: "all",
  lastPayload: null,
  batchPayload: null,
  selectedBatchVideo: null,
  batchFilter: "all",
  videoGridKey: "",
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

function formatNumber(value, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return value ?? "-";
  return number.toFixed(digits);
}

function formatFps(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return value ?? "-";
  return String(Math.round(number));
}

function formatBool(value) {
  if (value === true) return "是";
  if (value === false) return "否";
  return value ?? "-";
}

function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "-";
  if (value < 60) return `${Math.round(value)}s`;
  const minutes = Math.floor(value / 60);
  const rest = Math.round(value % 60);
  return `${minutes}m ${rest}s`;
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

// 数据规范化和流处理
function normalizeStreams(status) {
  if ((status.rtsp_summary || status.preview_video) && Array.isArray(status.streams) && status.streams.length > 0) {
    return status.streams;
  }

  const batchVideos = state.batchPayload?.videos || [];
  if (batchVideos.length > 0) {
    return batchVideos.map(batchVideoToStream);
  }

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
    note: "等待独立 OSD 推理视频",
  }));
}

function batchVideoToStream(video) {
  const stream = firstStream(video);
  const qualityStatus = video.quality?.quality_status || "unknown";
  const warning = batchMessages(video);
  return {
    name: `batch-${video.index}`,
    title: basename(video.input_video),
    source: video.input_video,
    kind: "file",
    source_type: "batch",
    state: qualityStatus === "failed" ? "ERROR" : qualityStatus === "review" ? "DEGRADED" : "RUNNING",
    inference_state: video.status || "unknown",
    playback: video.output_video_url || video.output_overlay_video_url || "",
    overlay: video.output_overlay_video_url || "",
    fps: formatFps(stream.estimated_fps),
    latency_ms: "-",
    detections: video.total_unique_persons ?? 0,
    dropped_frames: stream.is_frame_continuous === false ? 1 : 0,
    last_warning: qualityStatus === "review" ? warning || "需要复核" : null,
    last_error: qualityStatus === "failed" ? warning || "处理失败" : null,
    last_message_type: "BATCH_RESULT",
    note: "离线批量 OSD 输出",
    batchVideo: video,
  };
}

function selectedStream(streams) {
  if (state.selectedStream && !streams.some((item) => item.name === state.selectedStream)) {
    state.selectedStream = null;
  }
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
  const batchSummary = state.batchPayload?.summary || {};
  const batchQuality = state.batchPayload?.quality || {};
  const hasBatch = (state.batchPayload?.videos || []).length > 0;
  const hasRtsp = Boolean(status.rtsp_summary || status.preview_video);
  const alerts = streams.filter(hasAlert);
  const fpsValues = streams.map((item) => Number(item.fps)).filter((item) => Number.isFinite(item));
  const avgFps = fpsValues.length
    ? (fpsValues.reduce((sum, item) => sum + item, 0) / fpsValues.length).toFixed(1)
    : "-";
  const totalProcessingFps = Number(
    status.runtime_metrics?.processing_fps || status.rtsp_summary?.processing_fps || 0,
  );

  const pipelineState = hasBatch ? "BATCH_READY" : pipelineStatus.pipeline_state || "UNKNOWN";
  els.metricState.textContent = pipelineState;
  els.metricState.className =
    pipelineState === "PLAYING" || pipelineState === "BATCH_READY" || pipelineState === "READY" ? "metric-value ok" : "metric-value";
  els.metricStateSub.textContent = hasBatch
    ? `质量: ${batchQuality.passed_count ?? 0} passed`
    : hasRtsp
    ? `质量: ${status.rtsp_quality?.quality_status || "unknown"}`
    : `Bus: ${bus.last_message_type || "NONE"}`;

  els.metricStreams.textContent = String(hasBatch ? batchSummary.video_count ?? streams.length : status.source_count ?? streams.length);
  const runningCount = streams.filter((item) => item.state !== "ERROR").length;
  els.metricStreamsSub.textContent = hasBatch ? `${runningCount} 路验收通过/可查看` : `${runningCount} 路正常运行`;

  els.metricGpu.textContent = hasBatch ? "离线" : monitor.utilization_gpu == null ? "-" : `${monitor.utilization_gpu}%`;
  els.metricGpuSub.textContent = hasBatch
    ? "当前页面展示批量结果，不读取实时 GPU"
    : `内存 ${monitor.utilization_memory ?? "-"}% / 温度 ${monitor.temperature_c ?? "-"}°C`;

  els.metricFps.textContent = formatFps(totalProcessingFps || avgFps);
  els.metricFpsSub.textContent = hasBatch
    ? "批量结果时间轴 FPS"
    : hasRtsp
    ? `总处理 FPS / 单路均值: ${formatFps(totalProcessingFps)} / ${formatFps(avgFps)}`
    : `${status.controllers?.fps?.observations ?? 0} 次观测`;

  els.metricAlerts.textContent = String(alerts.length);
  els.metricAlerts.className = alerts.length > 0 ? "metric-value warn" : "metric-value ok";
  els.metricAlertsSub.textContent = alerts.length
    ? `${alerts[0].name}: ${alerts[0].last_warning || alerts[0].last_error || alerts[0].state}`
    : "系统运行正常";

  els.navAlertCount.textContent = String(alerts.length);
  els.busState.textContent = `Bus: ${bus.last_message_type || "NONE"}`;
  els.optimizationState.textContent = payload.optimization?.mode || "advisory_only";

  const filterLabel = state.filter === "alerts" ? "仅显示告警" : "显示全部";
  els.videoWallNote.textContent = hasBatch
    ? `${streams.length} 个批量输出 / ${filterLabel}`
    : hasRtsp
    ? `${streams.length} 路 RTSP health / 合成 OSD 在主视图播放`
    : `${streams.length} 路视频 / ${filterLabel}`;

  els.snapshotTime.textContent = formatTimestamp(payload.generated_at || status.app?.snapshot_at);
  state.lastUpdateTime = new Date();
}

// 渲染视频屏幕占位
function renderScreen(target, stream, large = false) {
  if (!stream) {
    target.innerHTML = '<div class="screen-center"><span>未选择视频流</span></div>';
    return;
  }

  const playable = large && stream.preview_playback ? stream.preview_playback : stream.playback || "";
  if (playable) {
    const title = escapeHtml(stream.title || stream.name);
    const source = escapeHtml(stream.source || "");
    const currentVideo = target.querySelector("video");
    if (!currentVideo || currentVideo.getAttribute("src") !== playable) {
      target.innerHTML = `
        <video controls preload="metadata" playsinline type="video/mp4" src="${escapeHtml(playable)}" style="width: 100%; height: 100%; min-height: ${large ? "420px" : "180px"}; object-fit: contain; background: #0f172a;"></video>
        <div class="screen-line" style="position: absolute; left: 16px; right: 16px; bottom: 16px; pointer-events: none;">
          <span class="badge" data-screen-title>${large && stream.preview_playback ? "独立 OSD 推理视频" : title}</span>
          <span class="badge" data-screen-source>${source}</span>
        </div>
      `;
    } else {
      const titleEl = target.querySelector("[data-screen-title]");
      const sourceEl = target.querySelector("[data-screen-source]");
      if (titleEl) titleEl.textContent = stream.title || stream.name || "";
      if (sourceEl) sourceEl.textContent = stream.source || "";
    }
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
  if (stream.preview_playback) {
    els.selectedSubtitle.textContent = `独立 OSD 推理视频 / ${stream.preview_playback}`;
  }
  els.selectedStatus.textContent = stream.state || "UNKNOWN";
  els.selectedStatus.className = `status ${statusClass(stream.state)}`;

  els.detailFps.textContent = formatFps(stream.fps);
  els.detailLatency.textContent = stream.latency_ms == null || stream.latency_ms === "-" ? "-" : `${stream.latency_ms} ms`;
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

  const gridKey = visibleStreams.map((stream) => `${stream.name}:${stream.playback || ""}`).join("|");
  if (state.videoGridKey !== gridKey) {
    state.videoGridKey = gridKey;
    els.videoGrid.innerHTML = visibleStreams
      .map((stream) => `
        <article class="video-tile" data-stream="${escapeHtml(stream.name)}">
          <div class="video-screen" data-screen="${escapeHtml(stream.name)}"></div>
          <div class="tile-meta">
            <strong data-tile-title="${escapeHtml(stream.name)}">${escapeHtml(stream.title || stream.name)}</strong>
            <span data-tile-status="${escapeHtml(stream.name)}" class="status">${escapeHtml(stream.state || "RUNNING")}</span>
          </div>
          <div class="tile-meta">
            <span data-tile-fps="${escapeHtml(stream.name)}"></span>
            <span data-tile-count="${escapeHtml(stream.name)}"></span>
          </div>
        </article>
      `)
      .join("");
  }

  document.querySelectorAll(".video-tile").forEach((tile) => {
    const name = tile.dataset.stream;
    const stream = visibleStreams.find((item) => item.name === name);
    if (!stream) return;
    tile.classList.toggle("selected", stream.name === state.selectedStream);
    const statusEl = tile.querySelector(`[data-tile-status="${CSS.escape(stream.name)}"]`);
    const fpsEl = tile.querySelector(`[data-tile-fps="${CSS.escape(stream.name)}"]`);
    const countEl = tile.querySelector(`[data-tile-count="${CSS.escape(stream.name)}"]`);
    if (statusEl) {
      statusEl.className = `status ${statusClass(stream.state)}`;
      statusEl.textContent = stream.state || "RUNNING";
    }
    if (fpsEl) fpsEl.textContent = `FPS: ${formatFps(stream.fps)} / 帧龄: ${stream.latency_ms ?? "-"} ms`;
    if (countEl) countEl.textContent = `检测: ${stream.detections ?? 0} / 丢帧: ${stream.dropped_frames ?? 0}`;
  });

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
        <strong>${formatFps(stream.fps)}</strong>
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
        <strong>当前 FPS:</strong> ${formatFps(stream.fps)}<br>
        <strong>最近帧年龄:</strong> ${stream.latency_ms ?? "-"} ms<br>
        <strong>最近帧检测对象数:</strong> ${stream.detections ?? 0}<br>
        <strong>累计丢帧数:</strong> ${stream.dropped_frames ?? 0}<br>
        <strong>估算丢帧率:</strong> ${((Number(stream.dropped_frame_rate || 0)) * 100).toFixed(2)}%
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
  renderBatchSummaryMetrics(summary, quality, videos);
  renderBatchArtifacts(payload.artifacts || {});

  if (!state.selectedBatchVideo && videos.length > 0) {
    state.selectedBatchVideo = String(videos[0].index || 1);
  }

  renderBatchTable(videos);
  renderBatchDetail(videos.find((video) => String(video.index) === state.selectedBatchVideo) || videos[0]);
  if (state.lastPayload) render(state.lastPayload);
}

function renderBatchSummaryMetrics(summary, quality, videos) {
  els.batchTotal.textContent = String(summary.video_count ?? videos.length ?? 0);
  els.batchOk.textContent = String(summary.processed_count ?? 0);
  els.batchReview.textContent = String(quality.review_count ?? 0);
  els.batchFailed.textContent = String(quality.failed_count ?? summary.failed_count ?? 0);
  els.batchPersons.textContent = String(summary.total_unique_persons_sum ?? 0);
  els.batchLine.textContent = `${summary.line_crossing_in_sum ?? 0} / ${summary.line_crossing_out_sum ?? 0}`;
  els.batchDuration.textContent = formatDuration(summary.total_duration_seconds);
  els.batchJobs.textContent = String(summary.batch_jobs ?? "-");
  els.batchProcessingFps.textContent = formatFps(summary.processing_fps);
  renderAcceptanceConclusion(summary, quality);
}

function renderRtspAcceptanceSummary(payload) {
  const summary = payload.summary || {};
  const quality = payload.quality || {};
  const runtimeMetrics = payload.runtime_metrics || {};
  const expected = Number(summary.expected_stream_count || 0);
  const observed = Number(summary.observed_stream_count || 0);
  const passed = Number(quality.passed_stream_count || 0);
  const review = Number(quality.review_stream_count || 0);
  const failed = Number(quality.failed_stream_count || 0);
  const status = quality.quality_status || "unknown";

  els.batchNote.textContent = payload.rtsp_dir
    ? `当前生产验收目录: ${payload.rtsp_dir}`
    : "尚未生成生产验收结果";
  els.batchTotal.textContent = String(expected || observed || 0);
  els.batchOk.textContent = String(passed);
  els.batchReview.textContent = String(review);
  els.batchFailed.textContent = String(failed);
  els.batchPersons.textContent = String(summary.total_unique_persons ?? 0);
  els.batchLine.textContent = `${summary.total_detections ?? 0} / ${summary.total_track_observations ?? 0}`;
  els.batchDuration.textContent = formatDuration(summary.run_seconds || runtimeMetrics.elapsed_seconds);
  els.batchJobs.textContent = String(summary.expected_stream_count ?? "-");
  els.batchProcessingFps.textContent = formatFps(summary.processing_fps || runtimeMetrics.processing_fps);
  els.acceptanceTitle.textContent =
    status === "passed" ? "本次生产验收通过" : status === "failed" ? "本次生产验收失败" : "本次生产验收需要复核";
  els.acceptanceDetail.textContent =
    `${observed}/${expected} 路 RTSP 已输出，帧数 ${summary.total_frame_count ?? 0}，检测 ${summary.total_detections ?? 0}，处理 FPS ${formatFps(summary.processing_fps || runtimeMetrics.processing_fps)}`;
  renderBatchArtifacts({
    summary: payload.artifacts?.summary,
    quality: payload.artifacts?.quality,
    html_report: payload.artifacts?.tiled_video,
    csv_report: payload.artifacts?.metrics_jsonl,
  });
}

function renderBatchArtifacts(artifacts) {
  const links = [
    ["batch_summary.json", artifacts.summary],
    ["batch_quality.json", artifacts.quality],
    ["HTML 报告", artifacts.html_report],
    ["CSV 报告", artifacts.csv_report],
  ]
    .filter(([, url]) => Boolean(url))
    .map(([label, url]) => `<a class="btn" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`)
    .join("");
  els.batchArtifacts.innerHTML = links || '<span class="empty-state">暂无验收报告文件</span>';
}

function renderBatchTable(videos) {
  const visible = videos.filter((video) => {
    const qualityStatus = video.quality?.quality_status || "unknown";
    const stream = firstStream(video);
    if (state.batchFilter === "all") return true;
    if (state.batchFilter === "review" || state.batchFilter === "failed") return qualityStatus === state.batchFilter;
    if (state.batchFilter === "with-person") return Number(video.total_unique_persons || 0) > 0;
    if (state.batchFilter === "no-person") return Number(video.total_unique_persons || 0) <= 0;
    if (state.batchFilter === "fps-low") return Number(stream.estimated_fps || 0) > 0 && Number(stream.estimated_fps || 0) < 20;
    if (state.batchFilter === "non-continuous") return stream.is_frame_continuous === false;
    return true;
  });

  if (!visible.length) {
    els.batchTableBody.innerHTML = '<tr><td colspan="11">当前筛选条件下没有批量结果</td></tr>';
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
          <td>${escapeHtml(formatFps(stream.estimated_fps))}</td>
          <td>${escapeHtml(formatFps(video.processing_fps))}</td>
          <td>${escapeHtml(formatBool(stream.is_frame_continuous))}</td>
          <td title="${escapeHtml(batchMessages(video))}">${escapeHtml(batchMessages(video) || "-")}</td>
        </tr>
      `;
    })
    .join("");
}

function renderBatchDetail(video) {
  const existingError = els.batchDetail.querySelector("[data-video-error]");
  if (existingError) existingError.remove();

  if (!video) {
    els.batchVideo.removeAttribute("src");
    els.batchDetail.innerHTML = '<div class="empty-state">未找到批量结果。先运行 scripts/legacy/person_analytics/run_person_analytics_batch.sh。</div>';
    return;
  }

  const stream = firstStream(video);
  const qualityStatus = video.quality?.quality_status || "unknown";
  const sizes = video.file_sizes || {};
  const playableUrl = video.output_video_url || video.output_overlay_video_url;
  if (playableUrl) {
    if (els.batchVideo.getAttribute("src") !== playableUrl) {
      els.batchVideo.setAttribute("src", playableUrl);
      els.batchVideo.load();
    }
  } else {
    els.batchVideo.removeAttribute("src");
  }

  const links = [
    ["播放视频", video.output_video_url],
    ["overlay文件", video.output_overlay_video_url],
    ["jsonl", video.output_jsonl_url],
    ["summary", video.output_summary_url],
    ["run.log", video.log_path_url],
  ]
    .filter(([, url]) => Boolean(url))
    .map(([label, url]) => `<a class="btn" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`)
    .join("");

  els.batchDetail.innerHTML = `
    <section class="batch-detail-section">
      <h3>当前视频验收</h3>
      <div class="batch-detail-grid">
        <div class="batch-detail-row"><span>视频</span><strong>${escapeHtml(basename(video.input_video))}</strong></div>
        <div class="batch-detail-row"><span>运行状态</span><strong>${escapeHtml(video.status || "unknown")}</strong></div>
        <div class="batch-detail-row"><span>质量状态</span><strong>${escapeHtml(qualityStatus)}</strong></div>
        <div class="batch-detail-row"><span>去重人数</span><strong>${escapeHtml(video.total_unique_persons ?? 0)}</strong></div>
        <div class="batch-detail-row"><span>ROI 人数</span><strong>${escapeHtml(formatRoi(video.roi_unique_persons))}</strong></div>
        <div class="batch-detail-row"><span>越线 In/Out</span><strong>${escapeHtml(video.line_crossing_in ?? 0)} / ${escapeHtml(video.line_crossing_out ?? 0)}</strong></div>
        <div class="batch-detail-row"><span>时间轴 FPS</span><strong>${escapeHtml(formatFps(stream.estimated_fps))}</strong></div>
        <div class="batch-detail-row"><span>处理 FPS</span><strong>${escapeHtml(formatFps(video.processing_fps))}</strong></div>
        <div class="batch-detail-row"><span>总帧数</span><strong>${escapeHtml(video.total_frame_count ?? stream.frame_count ?? 0)}</strong></div>
        <div class="batch-detail-row"><span>帧连续</span><strong>${escapeHtml(formatBool(stream.is_frame_continuous))}</strong></div>
        <div class="batch-detail-row"><span>原因</span><strong>${escapeHtml(batchMessages(video) || "无")}</strong></div>
      </div>
    </section>
    <section class="batch-detail-section">
      <h3>运行与输出</h3>
      <div class="batch-detail-grid">
        <div class="batch-detail-row"><span>运行耗时</span><strong>${escapeHtml(formatDuration(video.duration_seconds))}</strong></div>
        <div class="batch-detail-row"><span>输出大小</span><strong>${escapeHtml(formatBytes(sizes.output_video))}</strong></div>
        <div class="batch-detail-row"><span>Overlay 大小</span><strong>${escapeHtml(formatBytes(sizes.output_overlay_video))}</strong></div>
        <div class="batch-detail-row"><span>JSONL 大小</span><strong>${escapeHtml(formatBytes(sizes.output_jsonl))}</strong></div>
      </div>
      <div class="batch-links">${links || '<span class="empty-state">暂无输出链接</span>'}</div>
    </section>
    <section class="batch-detail-section wide">
      <h3>run.log 摘要</h3>
      <pre class="log-preview">${escapeHtml(video.log_tail || "暂无 run.log 摘要")}</pre>
    </section>
  `;
}

function renderAcceptanceConclusion(summary, quality) {
  const total = Number(summary.video_count || 0);
  const passed = Number(quality.passed_count || 0);
  const review = Number(quality.review_count || 0);
  const failed = Number(quality.failed_count || summary.failed_count || 0);
  if (!total) {
    els.acceptanceTitle.textContent = "等待验收结果";
    els.acceptanceDetail.textContent = "批量结果加载后显示本次验收结论";
    return;
  }
  if (failed > 0) {
    els.acceptanceTitle.textContent = "本次验收失败";
    els.acceptanceDetail.textContent = `${failed}/${total} 个视频失败，${review} 个需要复核`;
    return;
  }
  if (review > 0) {
    els.acceptanceTitle.textContent = "本次验收需要复核";
    els.acceptanceDetail.textContent = `${passed}/${total} 个通过，${review} 个需要人工复核`;
    return;
  }
  els.acceptanceTitle.textContent = "本次验收通过";
  els.acceptanceDetail.textContent = `${passed}/${total} 个视频 passed，总耗时 ${formatDuration(summary.total_duration_seconds)}，并发数 ${summary.batch_jobs ?? "-"}`;
}

function renderMultifileDashboard(payload) {
  const summary = payload.summary || {};
  const quality = payload.quality || {};
  const artifacts = payload.artifacts || {};
  const streams = summary.streams || {};
  const streamQuality = {};
  (quality.streams || []).forEach((item) => {
    streamQuality[item.stream_id] = item;
  });

  const pipelineDir = payload.rtsp_dir || payload.multifile_dir;
  els.multifileNote.textContent = pipelineDir
    ? `单 Pipeline 目录: ${pipelineDir}`
    : "尚未生成单 Pipeline 结果";
  els.multifileStatus.textContent = quality.quality_status || "unknown";
  els.multifileStatus.className = `status ${qualityClass(quality.quality_status || "unknown")}`;
  els.multifileStreams.textContent = `${summary.observed_stream_count ?? 0}/${summary.expected_stream_count ?? 0}`;
  els.multifileFrames.textContent = String(summary.total_frame_count ?? 0);
  els.multifileDetections.textContent = String(summary.total_detections ?? 0);
  els.multifilePersons.textContent = String(summary.total_unique_persons ?? 0);
  els.multifileReview.textContent = String(quality.review_stream_count ?? 0);
  els.multifileFailed.textContent = String(quality.failed_stream_count ?? 0);
  renderMultifileArtifacts(artifacts);
  renderIndividualVideos(artifacts.individual_outputs || []);
  renderMultifileTable(streams, streamQuality);
  renderMultifileDetail(payload);
  if (payload.rtsp_dir) {
    renderRtspAcceptanceSummary(payload);
  }

}

function renderMultifileArtifacts(artifacts) {
  const links = [
    ["独立视频索引", artifacts.individual_index],
    ["合并 MP4（可选）", artifacts.tiled_video],
    ["summary", artifacts.summary],
    ["quality", artifacts.quality],
    ["jsonl", artifacts.jsonl],
    ["metrics", artifacts.metrics_jsonl],
    ["recovery", artifacts.recovery_check],
    ["run.log", artifacts.run_log],
  ]
    .filter(([, url]) => Boolean(url))
    .map(([label, url]) => `<a class="btn" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`)
    .join("");
  els.multifileArtifacts.innerHTML = links || '<span class="empty-state">暂无单 Pipeline 输出链接</span>';
}

function renderIndividualVideos(outputs) {
  if (!els.multifileIndividualVideos) return;
  if (!Array.isArray(outputs) || outputs.length === 0) {
    els.multifileIndividualVideos.innerHTML = '<div class="empty-state">暂无独立推理视频，请先完成带 ENABLE_INDIVIDUAL_OUTPUTS 的验收运行</div>';
    return;
  }
  els.multifileIndividualVideos.innerHTML = outputs
    .map((item) => `
      <article class="individual-video-card">
        <div class="individual-video-head">
          <strong>${escapeHtml(item.stream_id || "unknown")}</strong>
          <a href="${escapeHtml(item.video_url)}" target="_blank" rel="noreferrer">打开原视频</a>
        </div>
        <video controls preload="metadata" playsinline type="video/mp4" src="${escapeHtml(item.video_url)}"></video>
      </article>
    `)
    .join("");
}

function renderMultifileTable(streams, streamQuality) {
  const entries = Object.entries(streams || {}).sort(([left], [right]) => left.localeCompare(right));
  if (!entries.length) {
    els.multifileTableBody.innerHTML = '<tr><td colspan="9">单 Pipeline 结果未就绪</td></tr>';
    return;
  }
  els.multifileTableBody.innerHTML = entries
    .map(([streamId, stream]) => {
      const quality = streamQuality[streamId] || {};
      const messages = [...(quality.failures || []), ...(quality.reviews || [])].join("; ");
      return `
        <tr>
          <td>${escapeHtml(streamId)}</td>
          <td><span class="status ${qualityClass(quality.quality_status || "unknown")}">${escapeHtml(quality.quality_status || "unknown")}</span></td>
          <td>${escapeHtml(stream.frame_count ?? 0)}</td>
          <td>${escapeHtml(stream.total_detections ?? 0)}</td>
          <td>${escapeHtml(stream.total_track_observations ?? 0)}</td>
          <td>${escapeHtml(stream.total_unique_persons ?? 0)}</td>
          <td>${escapeHtml(formatFps(stream.estimated_fps))}</td>
          <td>${escapeHtml(formatBool(stream.is_frame_continuous))}</td>
          <td title="${escapeHtml(messages)}">${escapeHtml(messages || "-")}</td>
        </tr>
      `;
    })
    .join("");
}

function renderMultifileDetail(payload) {
  const quality = payload.quality || {};
  const summary = payload.summary || {};
  const runtimeMetrics = payload.runtime_metrics || {};
  const sourceStatus = payload.source_status || summary.source_status_payload || {};
  const logTail = payload.log_tail || "";
  const messages = [...(quality.failures || []), ...(quality.reviews || [])];
  els.multifileDetail.innerHTML = `
    <section class="batch-detail-section">
      <h3>单 Pipeline 验收</h3>
      <div class="batch-detail-grid">
        <div class="batch-detail-row"><span>模式</span><strong>${escapeHtml(summary.mode || "inprocess_multifile")}</strong></div>
        <div class="batch-detail-row"><span>RTSP Base</span><strong>${escapeHtml(summary.rtsp_base || "-")}</strong></div>
        <div class="batch-detail-row"><span>质量状态</span><strong>${escapeHtml(quality.quality_status || "unknown")}</strong></div>
        <div class="batch-detail-row"><span>缺失路数</span><strong>${escapeHtml((summary.missing_stream_ids || []).join(", ") || "无")}</strong></div>
        <div class="batch-detail-row"><span>原因</span><strong>${escapeHtml(messages.join("; ") || "无")}</strong></div>
      </div>
    </section>
    <section class="batch-detail-section wide">
      <h3>Runtime Metrics</h3>
      <div class="batch-detail-grid">
        <div class="batch-detail-row"><span>采样时间</span><strong>${escapeHtml(formatTimestamp(runtimeMetrics.timestamp))}</strong></div>
        <div class="batch-detail-row"><span>运行时长</span><strong>${escapeHtml(formatDuration(runtimeMetrics.elapsed_seconds))}</strong></div>
        <div class="batch-detail-row"><span>总处理 FPS</span><strong>${escapeHtml(formatNumber(runtimeMetrics.processing_fps, 1))}</strong></div>
        <div class="batch-detail-row"><span>总帧数</span><strong>${escapeHtml(runtimeMetrics.total_frames ?? "-")}</strong></div>
        <div class="batch-detail-row"><span>进程内存</span><strong>${escapeHtml(formatBytes((runtimeMetrics.process?.max_rss_kb || 0) * 1024))}</strong></div>
        <div class="batch-detail-row"><span>CPU 时间</span><strong>${escapeHtml(formatNumber((runtimeMetrics.process?.user_cpu_seconds || 0) + (runtimeMetrics.process?.system_cpu_seconds || 0), 1))}s</strong></div>
      </div>
    </section>
    <section class="batch-detail-section wide">
      <h3>Source Health</h3>
      ${renderSourceHealthTable(runtimeMetrics.streams || {}, sourceStatus.streams || [])}
    </section>
    <section class="batch-detail-section wide">
      <h3>run.log 摘要</h3>
      <pre class="log-preview">${escapeHtml(logTail || "暂无 run.log 摘要")}</pre>
    </section>
  `;
}

function renderSourceHealthTable(runtimeStreams, sourceStreams) {
  const sourceById = {};
  (sourceStreams || []).forEach((item) => {
    const index = Number(item.index);
    if (Number.isFinite(index) && index > 0) {
      sourceById[`stream-${index - 1}`] = item;
    } else {
      sourceById[item.stream_id] = item;
    }
  });
  const streamIds = Array.from(new Set([
    ...Object.keys(runtimeStreams || {}),
    ...Object.keys(sourceById),
  ])).sort();
  if (!streamIds.length) {
    return '<div class="empty-state">暂无 runtime_metrics/source_status 数据</div>';
  }
  const rows = streamIds.map((streamId) => {
    const runtime = runtimeStreams[streamId] || {};
    const source = sourceById[streamId] || {};
    const status = runtime.status || source.status || "unknown";
    return `
      <tr>
        <td>${escapeHtml(streamId)}</td>
        <td><span class="status ${qualityClass(status === "online" ? "passed" : status === "stale" ? "review" : "failed")}">${escapeHtml(status)}</span></td>
        <td>${escapeHtml(runtime.frame_count ?? "-")}</td>
        <td>${escapeHtml(runtime.last_frame_id ?? "-")}</td>
        <td>${escapeHtml(formatNumber(runtime.last_seen_age_seconds, 1))}</td>
        <td>${escapeHtml(formatFps(runtime.estimated_processing_fps))}</td>
        <td>${escapeHtml(runtime.stale_count ?? 0)}</td>
        <td>${escapeHtml(runtime.recovered_count ?? 0)}</td>
        <td>${escapeHtml(runtime.keepalive_active ? "是" : "否")}</td>
        <td>${escapeHtml(source.pid ?? "-")}</td>
        <td>${escapeHtml(source.restart_count ?? 0)}</td>
        <td title="${escapeHtml(source.last_error || "")}">${escapeHtml(source.last_error || "-")}</td>
      </tr>
    `;
  }).join("");
  return `
    <div class="batch-table-wrap">
      <table class="batch-table">
        <thead>
          <tr>
            <th>stream</th>
            <th>健康</th>
            <th>帧数</th>
            <th>最后帧</th>
            <th>无帧秒数</th>
            <th>处理 FPS</th>
            <th>stale</th>
            <th>recovered</th>
            <th>保活</th>
            <th>pid</th>
            <th>restart</th>
            <th>source error</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
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
    els.batchTableBody.innerHTML = `<tr><td colspan="11">批量结果未就绪：${escapeHtml(error.message)}</td></tr>`;
    els.batchDetail.innerHTML = '<div class="empty-state">先运行批量处理，生成 outputs/batch/batch_summary.json 和 batch_quality.json。</div>';
  }
}

async function refreshMultifileDashboard() {
  try {
    let response = await fetch("/api/rtsp/dashboard");
    if (!response.ok) {
      response = await fetch("/api/multifile/dashboard");
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const payload = await response.json();
    renderMultifileDashboard(payload);
  } catch (error) {
    console.warn("加载单 Pipeline 结果失败:", error);
    els.multifileTableBody.innerHTML = `<tr><td colspan="9">单 Pipeline 结果未就绪：${escapeHtml(error.message)}</td></tr>`;
    els.multifileDetail.innerHTML = '<div class="empty-state">先运行 scripts/legacy/person_analytics/run_multifile_inproc.sh，生成 multifile_summary.json 和 multifile_quality.json。</div>';
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
refreshMultifileDashboard();
refreshBatchDashboard();
window.setInterval(refresh, 10000);
window.setInterval(refreshMultifileDashboard, 5000);
window.setInterval(refreshBatchDashboard, 5000);

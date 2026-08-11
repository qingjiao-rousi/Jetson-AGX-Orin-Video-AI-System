#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "index",
    "input_video",
    "status",
    "total_unique_persons",
    "roi_unique_persons",
    "line_crossing_in",
    "line_crossing_out",
    "frame_count",
    "estimated_fps",
    "is_frame_continuous",
    "output_video",
    "output_overlay_video",
    "output_jsonl",
    "output_summary",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CSV and HTML reports from batch_summary.json.")
    parser.add_argument("batch_summary_json", type=Path, help="Input batch_summary.json path.")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Output directory. Defaults to the batch_summary.json parent directory.",
    )
    parser.add_argument("--csv-name", default="batch_summary.csv", help="CSV output file name.")
    parser.add_argument("--html-name", default="batch_report.html", help="HTML output file name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.batch_summary_json.parent
    csv_path = output_dir / args.csv_name
    html_path = output_dir / args.html_name
    export_report(args.batch_summary_json, csv_path, html_path)
    print(f"Wrote CSV report: {csv_path}")
    print(f"Wrote HTML report: {html_path}")
    return 0


def export_report(batch_summary_json: Path, csv_path: Path, html_path: Path) -> None:
    summary = _read_json(batch_summary_json)
    rows = build_rows(summary)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, csv_path)
    write_html(summary, rows, html_path)


def build_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, video in enumerate(summary.get("videos", []), start=1):
        stream = _first_stream(video.get("streams", {}))
        rows.append(
            {
                "index": index,
                "input_video": video.get("input_video", ""),
                "status": video.get("status", "unknown"),
                "total_unique_persons": video.get("total_unique_persons", ""),
                "roi_unique_persons": _format_roi(video.get("roi_unique_persons", {})),
                "line_crossing_in": video.get("line_crossing_in", ""),
                "line_crossing_out": video.get("line_crossing_out", ""),
                "frame_count": stream.get("frame_count", ""),
                "estimated_fps": stream.get("estimated_fps", ""),
                "is_frame_continuous": stream.get("is_frame_continuous", ""),
                "output_video": video.get("output_video", ""),
                "output_overlay_video": video.get("output_overlay_video", ""),
                "output_jsonl": video.get("output_jsonl", ""),
                "output_summary": video.get("output_summary", ""),
                "error": video.get("error", ""),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def write_html(summary: dict[str, Any], rows: list[dict[str, Any]], html_path: Path) -> None:
    html_path.write_text(_render_html(summary, rows, html_path), encoding="utf-8")


def _render_html(summary: dict[str, Any], rows: list[dict[str, Any]], html_path: Path) -> str:
    cards = [
        ("视频总数", summary.get("video_count", 0)),
        ("成功", summary.get("processed_count", 0)),
        ("失败", summary.get("failed_count", 0)),
        ("去重人数合计", summary.get("total_unique_persons_sum", 0)),
        ("越线 In", summary.get("line_crossing_in_sum", 0)),
        ("越线 Out", summary.get("line_crossing_out_sum", 0)),
    ]
    card_html = "\n".join(
        f'<section class="metric"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></section>'
        for label, value in cards
    )
    row_html = "\n".join(_render_row(row, html_path) for row in rows)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Person Batch Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #607080;
      --line: #d9e0e7;
      --ok: #0f766e;
      --bad: #b42318;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 720;
      letter-spacing: 0;
    }}
    .sub {{
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 14px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .metric strong {{
      display: block;
      font-size: 24px;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      min-width: 1120px;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 680;
      background: #fbfcfd;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .status {{
      display: inline-block;
      min-width: 56px;
      border-radius: 999px;
      padding: 2px 8px;
      text-align: center;
      font-weight: 650;
      font-size: 12px;
    }}
    .status.ok {{ background: #dff7f2; color: var(--ok); }}
    .status.failed {{ background: #fee4e2; color: var(--bad); }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .muted {{ color: var(--muted); }}
    .error {{ color: var(--bad); max-width: 260px; white-space: normal; }}
    @media (max-width: 900px) {{
      main {{ padding: 20px 12px 32px; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Person Batch Report</h1>
    <p class="sub">Batch directory: {_escape(summary.get("batch_dir", ""))}</p>
    <div class="metrics">
      {card_html}
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>输入视频</th>
            <th>状态</th>
            <th>人数</th>
            <th>ROI</th>
            <th>越线 In/Out</th>
            <th>帧数</th>
            <th>FPS</th>
            <th>连续</th>
            <th>输出</th>
            <th>错误</th>
          </tr>
        </thead>
        <tbody>
          {row_html}
        </tbody>
      </table>
    </div>
  </main>
</body>
</html>
"""


def _render_row(row: dict[str, Any], html_path: Path) -> str:
    status = str(row.get("status", "unknown"))
    status_class = "ok" if status == "ok" else "failed"
    input_name = Path(str(row.get("input_video", ""))).name
    video_link = _make_link(row.get("output_video", ""), "video", html_path)
    overlay_link = _make_link(row.get("output_overlay_video", ""), "overlay", html_path)
    jsonl_link = _make_link(row.get("output_jsonl", ""), "jsonl", html_path)
    summary_link = _make_link(row.get("output_summary", ""), "summary", html_path)
    return f"""<tr>
  <td>{_escape(row.get("index", ""))}</td>
  <td title="{_escape(row.get("input_video", ""))}">{_escape(input_name)}</td>
  <td><span class="status {status_class}">{_escape(status)}</span></td>
  <td>{_escape(row.get("total_unique_persons", ""))}</td>
  <td>{_escape(row.get("roi_unique_persons", ""))}</td>
  <td>{_escape(row.get("line_crossing_in", ""))} / {_escape(row.get("line_crossing_out", ""))}</td>
  <td>{_escape(row.get("frame_count", ""))}</td>
  <td>{_escape(_format_float(row.get("estimated_fps", "")))}</td>
  <td>{_escape(row.get("is_frame_continuous", ""))}</td>
  <td>{video_link} {overlay_link} {jsonl_link} {summary_link}</td>
  <td class="error">{_escape(row.get("error", ""))}</td>
</tr>"""


def _first_stream(streams: Any) -> dict[str, Any]:
    if not isinstance(streams, dict) or not streams:
        return {}
    first_key = sorted(streams.keys())[0]
    stream = streams.get(first_key, {})
    return stream if isinstance(stream, dict) else {}


def _format_roi(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return "; ".join(f"{key}={value[key]}" for key in sorted(value))


def _format_float(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.2f}"
    return value


def _make_link(raw_path: Any, label: str, html_path: Path) -> str:
    if not raw_path:
        return f'<span class="muted">{_escape(label)}</span>'
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = Path.cwd() / path
    target = os.path.relpath(path, start=html_path.parent)
    return f'<a href="{html.escape(target, quote=True)}">{_escape(label)}</a>'


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())

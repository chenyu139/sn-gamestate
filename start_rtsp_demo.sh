#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_PATH="$ROOT_DIR/data/football.mp4"
MEDIAMTX_BIN="/home/chenyu/miniconda3/bin/mediamtx"
BIND_HOST="0.0.0.0"
RTSP_PORT="9554"
RTP_PORT="9000"
RTCP_PORT="9001"
RTMP_PORT="2935"
HLS_PORT="9888"
WEBRTC_PORT="9889"
WEBRTC_UDP_PORT="9189"
SRT_PORT="9890"
INPUT_STREAM_NAME="football"
OUTPUT_STREAM_NAME="football_tracked"
RUN_DIR="/tmp/sn_rtsp_demo"
LOG_DIR="$RUN_DIR/logs"
INPUT_PUSH_URL="rtsp://127.0.0.1:${RTSP_PORT}/${INPUT_STREAM_NAME}"
OUTPUT_PUSH_URL="rtsp://127.0.0.1:${RTSP_PORT}/${OUTPUT_STREAM_NAME}"
WSL_IP="$(hostname -I | awk '{print $1}')"

mkdir -p "$LOG_DIR"
: >"$LOG_DIR/mediamtx.log"
: >"$LOG_DIR/source.log"
: >"$LOG_DIR/pipeline.log"

for command_name in ffmpeg uv; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "缺少命令: $command_name" >&2
    exit 1
  fi
done

if [ ! -x "$MEDIAMTX_BIN" ]; then
  echo "MediaMTX 不存在或不可执行: $MEDIAMTX_BIN" >&2
  exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
  echo "视频文件不存在: $VIDEO_PATH" >&2
  exit 1
fi

collect_busy_pids() {
  {
    ss -ltnp "( sport = :${RTSP_PORT} or sport = :${RTMP_PORT} or sport = :${HLS_PORT} or sport = :${WEBRTC_PORT} )" 2>/dev/null || true
    ss -lunp "( sport = :${RTP_PORT} or sport = :${RTCP_PORT} or sport = :${WEBRTC_UDP_PORT} or sport = :${SRT_PORT} )" 2>/dev/null || true
  } | grep -o 'pid=[0-9]\+' | cut -d= -f2 | sort -u || true
}

busy_pids="$(collect_busy_pids)"
if [ -n "$busy_pids" ]; then
  echo "检测到相关端口被占用，正在清理 PID: $(echo "$busy_pids" | tr '\n' ' ' | xargs)"
  while IFS= read -r pid; do
    if [ -n "$pid" ]; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done <<<"$busy_pids"
  fuser -k 9554/tcp 9000/udp 9001/udp 2935/tcp 9888/tcp 9889/tcp 9189/udp 9890/udp >/dev/null 2>&1 || true
  sleep 2
fi

busy_pids="$(collect_busy_pids)"
if [ -n "$busy_pids" ]; then
  echo "端口清理失败，仍有 PID 占用: $(echo "$busy_pids" | tr '\n' ' ' | xargs)" >&2
  exit 1
fi

cleanup() {
  local exit_code=$?
  set +e
  for pid in "${PIPELINE_PID:-}" "${SOURCE_PID:-}" "${MEDIAMTX_PID:-}"; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1
      wait "$pid" >/dev/null 2>&1
    fi
  done
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "日志目录: $LOG_DIR"
echo "启动 MediaMTX..."
"$MEDIAMTX_BIN" "$ROOT_DIR/mediamtx.local.yml" >"$LOG_DIR/mediamtx.log" 2>&1 &
MEDIAMTX_PID=$!
sleep 2

if ! kill -0 "$MEDIAMTX_PID" >/dev/null 2>&1; then
  echo "MediaMTX 启动失败" >&2
  tail -n 50 "$LOG_DIR/mediamtx.log" >&2 || true
  exit 1
fi

echo "启动 football.mp4 -> RTSP 实时模拟源..."
ffmpeg -re -stream_loop -1 -i "$VIDEO_PATH" -an -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -rtsp_transport tcp -f rtsp "$INPUT_PUSH_URL" >"$LOG_DIR/source.log" 2>&1 &
SOURCE_PID=$!
sleep 4

if ! kill -0 "$SOURCE_PID" >/dev/null 2>&1; then
  echo "输入 RTSP 模拟源启动失败" >&2
  tail -n 50 "$LOG_DIR/source.log" >&2 || true
  exit 1
fi

echo "启动实时跟踪 pipeline..."
LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" MPLCONFIGDIR=/tmp/mpl uv run python -m sn_gamestate.live_main dataset.source="$INPUT_PUSH_URL" visualization.save_video=False visualization.rtsp_url="$OUTPUT_PUSH_URL" engine.max_frames=-1 engine.read_timeout_ms=5000 hydra.run.dir="$RUN_DIR/hydra" >"$LOG_DIR/pipeline.log" 2>&1 &
PIPELINE_PID=$!
sleep 6

if ! kill -0 "$PIPELINE_PID" >/dev/null 2>&1; then
  echo "实时跟踪 pipeline 启动失败" >&2
  tail -n 80 "$LOG_DIR/pipeline.log" >&2 || true
  exit 1
fi

echo "已启动完成"
echo "WSL 内访问:"
echo "  输入流:  rtsp://127.0.0.1:${RTSP_PORT}/${INPUT_STREAM_NAME}"
echo "  输出流:  rtsp://127.0.0.1:${RTSP_PORT}/${OUTPUT_STREAM_NAME}"
if [ -n "$WSL_IP" ]; then
  echo "Windows 访问:"
  echo "  输入流:  rtsp://${WSL_IP}:${RTSP_PORT}/${INPUT_STREAM_NAME}"
  echo "  输出流:  rtsp://${WSL_IP}:${RTSP_PORT}/${OUTPUT_STREAM_NAME}"
fi
echo "日志文件:"
echo "  $LOG_DIR/mediamtx.log"
echo "  $LOG_DIR/source.log"
echo "  $LOG_DIR/pipeline.log"
echo "按 Ctrl+C 停止全部进程"

while true; do
  if ! kill -0 "$MEDIAMTX_PID" >/dev/null 2>&1; then
    echo "MediaMTX 已退出" >&2
    tail -n 50 "$LOG_DIR/mediamtx.log" >&2 || true
    exit 1
  fi
  if ! kill -0 "$SOURCE_PID" >/dev/null 2>&1; then
    echo "RTSP 模拟源已退出" >&2
    tail -n 50 "$LOG_DIR/source.log" >&2 || true
    exit 1
  fi
  if ! kill -0 "$PIPELINE_PID" >/dev/null 2>&1; then
    echo "实时跟踪 pipeline 已退出" >&2
    tail -n 80 "$LOG_DIR/pipeline.log" >&2 || true
    exit 1
  fi
  sleep 2
done

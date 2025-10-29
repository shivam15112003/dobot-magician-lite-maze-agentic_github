#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -d "$HERE/.venv" ]; then python3 -m venv "$HERE/.venv"; fi
source "$HERE/.venv/bin/activate"
python -m pip install --upgrade pip
pip install -r "$HERE/requirements.txt"
IMAGE="${IMAGE:-$HERE/maze8.jpg}"
PORT="${1:-auto}"
START_COLOR="${START_COLOR:-red}"
ANIMATE="${ANIMATE:-0}"
SCENE_AGENT="${SCENE_AGENT:-gemini}"
CAM_INDEX="${CAMERA_INDEX:-}"
CAM_W="${CAMERA_W:-}"
CAM_H="${CAMERA_H:-}"
CAM_AUTO="${CAMERA_AUTO_CAPTURE:-0}"
EXTRA_ARGS=""; if [ "$ANIMATE" = "1" ]; then EXTRA_ARGS="--animate"; fi
if [ -n "$CAM_INDEX" ]; then
  CMD=(python "$HERE/main.py" --camera-index "$CAM_INDEX" --start-color "$START_COLOR" --scene-agent "$SCENE_AGENT" $EXTRA_ARGS)
  if [ -n "$CAM_W" ]; then CMD+=("--camera-w" "$CAM_W"); fi
  if [ -n "$CAM_H" ]; then CMD+=("--camera-h" "$CAM_H"); fi
  if [ "$CAM_AUTO" = "1" ]; then CMD+=("--camera-auto-capture" "--no-show"); fi
  "${CMD[@]}"
else
  if [ "$PORT" = "auto" ] || [ -z "$PORT" ]; then
    python "$HERE/main.py" --image "$IMAGE" --start-color "$START_COLOR" --scene-agent "$SCENE_AGENT" $EXTRA_ARGS
  else
    python "$HERE/main.py" --image "$IMAGE" --port "$PORT" --start-color "$START_COLOR" --scene-agent "$SCENE_AGENT" $EXTRA_ARGS
  fi
fi

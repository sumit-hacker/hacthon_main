#!/usr/bin/env bash
set -euo pipefail

# Regular daily start (no dependency install)
# Assumes:
# - backend/.venv exists and dependencies are installed
# - frontend/node_modules exists
# - backend/.env exists

BACKEND_PORT="8000"
FRONTEND_PORT="5173"

usage() {
  cat <<'EOF'
Usage:
  ./start.sh [--backend-port 8000] [--frontend-port 5173]

This does NOT install dependencies.
If you are running for the first time, use:
  ./run.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend-port)
      BACKEND_PORT="${2:-}"; shift 2 ;;
    --frontend-port)
      FRONTEND_PORT="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

if [[ ! -d "$BACKEND_DIR/.venv" ]]; then
  echo "Missing backend/.venv. Run ./run.sh first." >&2
  exit 1
fi

if [[ ! -f "$BACKEND_DIR/.env" ]]; then
  echo "Missing backend/.env. Run ./run.sh first (it copies .env.example)." >&2
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Missing frontend/node_modules. Run ./run.sh or run: (cd frontend && npm ci)" >&2
  exit 1
fi

cleanup() {
  echo "\nStopping servers…"
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

(
  cd "$BACKEND_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload
) &
BACKEND_PID=$!

(
  cd "$FRONTEND_DIR"
  exec npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

echo "Backend:  http://localhost:${BACKEND_PORT}/api/health"
echo "Frontend: http://localhost:${FRONTEND_PORT}"
echo "\nServers are running. Keep this terminal open. Press Ctrl+C to stop."

wait

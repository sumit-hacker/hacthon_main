#!/usr/bin/env bash
set -euo pipefail

# First-time setup + start
# - Creates backend venv and installs Python deps
# - Installs frontend deps via npm ci
# - Ensures backend/.env exists (copies from .env.example if missing)
# - Optionally sets RAG_MODE (local/online/hybrid)
# - Starts backend + frontend together

MODE="local"
BACKEND_PORT="8000"
FRONTEND_PORT="5173"
USE_LOCK="0"
TORCH_CPU="1"
TORCH_VERSION="2.10.0"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh [--mode local|online|hybrid] [--backend-port 8000] [--frontend-port 5173] [--use-lock]

Examples:
  ./run.sh --mode local
  ./run.sh --mode online
  ./run.sh --mode hybrid
  ./run.sh --mode local --use-lock

Notes:
- Local mode requires no API keys, but may download model weights on first warmup.
- Online/hybrid modes require you to set keys in backend/.env.
- By default this script installs Python deps from backend/requirements.txt for portability.
- The lockfile backend/requirements.lock.txt may include CUDA (very large). Use --use-lock only if you really want that exact env.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --backend-port)
      BACKEND_PORT="${2:-}"; shift 2 ;;
    --frontend-port)
      FRONTEND_PORT="${2:-}"; shift 2 ;;
    --use-lock)
      USE_LOCK="1"; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "local" && "$MODE" != "online" && "$MODE" != "hybrid" ]]; then
  echo "Invalid --mode: $MODE (expected local|online|hybrid)" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

if [[ ! -d "$BACKEND_DIR" || ! -d "$FRONTEND_DIR" ]]; then
  echo "Expected backend/ and frontend/ folders next to this script." >&2
  exit 1
fi

command -v python >/dev/null 2>&1 || { echo "python not found" >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm not found" >&2; exit 1; }

echo "==> [1/5] Backend: creating venv + installing deps"
cd "$BACKEND_DIR"

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# Install CPU-only PyTorch first to avoid pulling multi-GB CUDA wheels in environments
# without GPUs (common in Codespaces). Requirements will consider this satisfied.
if [[ "$TORCH_CPU" == "1" ]]; then
  python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==${TORCH_VERSION}"
fi

if [[ "$USE_LOCK" == "1" && -f requirements.lock.txt ]]; then
  python -m pip install -r requirements.lock.txt
else
  python -m pip install -r requirements.txt
fi

echo "==> [2/5] Backend: ensuring .env exists and setting mode=$MODE"
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
  else
    echo "backend/.env.example missing; cannot create backend/.env" >&2
    exit 1
  fi
fi

# Set RAG_MODE in .env (create or replace the line)
if grep -qE '^RAG_MODE=' .env; then
  sed -i -E "s/^RAG_MODE=.*/RAG_MODE=$MODE/" .env
else
  printf "\nRAG_MODE=%s\n" "$MODE" >> .env
fi

if [[ "$MODE" != "local" ]]; then
  echo "NOTE: You selected $MODE mode. Edit backend/.env and set API keys (Azure/OpenAI/Qdrant/Mongo) as needed."
fi

echo "==> [3/5] Frontend: installing deps (npm ci)"
cd "$FRONTEND_DIR"
# npm ci requires package-lock.json; fall back to npm install if missing.
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi

echo "==> [4/5] Starting backend + frontend"
cd "$ROOT_DIR"

cleanup() {
  echo "\nStopping servers…"
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Start backend
(
  cd "$BACKEND_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload
) &
BACKEND_PID=$!

# Start frontend
(
  cd "$FRONTEND_DIR"
  exec npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

sleep 1

echo "==> [5/5] Quick health checks"
BACKEND_HEALTH="http://localhost:${BACKEND_PORT}/api/health"
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"

echo "Backend:  $BACKEND_HEALTH"
echo "Frontend: $FRONTEND_URL"

if command -v curl >/dev/null 2>&1; then
  curl -sS --max-time 5 "$BACKEND_HEALTH" >/dev/null && echo "Backend health: OK" || echo "Backend health: NOT READY YET"
fi

echo "\nServers are running. Keep this terminal open. Press Ctrl+C to stop."
wait

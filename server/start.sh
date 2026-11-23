#!/bin/sh
set -eu

WORKER_PORT="${DOTNXS_WORKER_PORT:-8001}"
python -m uvicorn server.worker:app --host 127.0.0.1 --port "$WORKER_PORT" --workers 1 &
worker_pid=$!
/usr/local/bin/dotnxs-gateway &
gateway_pid=$!

cleanup() {
  kill -TERM "$gateway_pid" "$worker_pid" 2>/dev/null || true
  wait "$gateway_pid" 2>/dev/null || true
  wait "$worker_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait "$gateway_pid"

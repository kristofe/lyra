#!/usr/bin/env bash
#
# start_server.sh — launch the Lyra-2 inference API (terminal 1 of 2).
#
# Usage:
#   Terminal 1:  ./start_server.sh        # leave this running
#   Terminal 2:  ./run_sequential_gens.sh # once you see "Application startup complete"
#
# This generates a fresh bearer token, writes it to <repo>/.lyra_token so the
# client script can read it, then starts the server on :8000. The server enforces
# the token on every endpoint except /health. Ctrl-C stops it.
set -euo pipefail

REPO="${LYRA_REPO:-/teamspace/studios/this_studio/world_model/lyra2}"
cd "$REPO"

# Stable bearer token, shared with run_sequential_gens.sh via this file.
# Precedence: an already-exported $LYRA_API_TOKEN wins; otherwise reuse the token
# saved from a previous run; otherwise mint a fresh one. Delete .lyra_token (or
# export a new LYRA_API_TOKEN) to rotate it.
TOKEN_FILE="$REPO/.lyra_token"
if [ -z "${LYRA_API_TOKEN:-}" ]; then
  if [ -s "$TOKEN_FILE" ]; then
    LYRA_API_TOKEN="$(cat "$TOKEN_FILE")"
  else
    LYRA_API_TOKEN="$(openssl rand -hex 32)"
  fi
fi
export LYRA_API_TOKEN
echo "$LYRA_API_TOKEN" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"

echo "[start_server] repo:  $REPO"
echo "[start_server] token: written to $REPO/.lyra_token (run_sequential_gens.sh reads it)"
echo "[start_server] starting API on 0.0.0.0:8000 — wait for 'Application startup complete' (~80s)."
echo "[start_server] then run ./run_sequential_gens.sh in a second terminal. Ctrl-C here to stop."
echo

# run_server_cu12.sh sets PYTHONPATH/CUDA_HOME/LD_LIBRARY_PATH itself and execs uvicorn.
exec bash api/run_server_cu12.sh

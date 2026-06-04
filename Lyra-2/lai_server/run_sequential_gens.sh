#!/usr/bin/env bash
#
# run_sequential_gens.sh — drive two sequential generations against the running API
# (terminal 2 of 2). Demonstrates that spatial memory accumulates across a session.
#
# Prereq: ./start_server.sh is running in another terminal.
#
#   Gen 0 — new session, FORWARD one chunk      (seeds "point A" from the museum image)
#   Gen 1 — continue session, ROTATE LEFT chunk (reuses point A + accumulated memory)
#
# Both are 240p (240x416), 81 frames = one chunk. Outputs land in
# <repo>/outputs/sequential_test/. A growing X-Cache-Entries (~11 -> ~21) is the
# signal that the spatial memory persisted and reloaded between the two calls.
set -euo pipefail

REPO="${LYRA_REPO:-/teamspace/studios/this_studio/world_model/lyra2}"
cd "$REPO"

URL="${URL:-http://localhost:8000}"
TOKEN_FILE="$REPO/.lyra_token"
PROMPT="A high-quality, photorealistic scene with smooth, stable camera motion."
IMAGE="$REPO/assets/samples_custom/museum/image.png"
TRAJ="$REPO/assets/trajectories"
OUT="$REPO/outputs/sequential_test"
mkdir -p "$OUT"

# --- token (written by start_server.sh) ---
if [ ! -s "$TOKEN_FILE" ]; then
  echo "ERROR: $TOKEN_FILE not found. Start the server first in another terminal:" >&2
  echo "       ./start_server.sh" >&2
  exit 1
fi
TOKEN="$(cat "$TOKEN_FILE")"

# --- wait for the server to be ready (covers cold start) ---
echo "[gens] waiting for $URL/health ..."
ready=0
for _ in $(seq 1 120); do
  if curl -sf "$URL/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 3
done
if [ "$ready" -ne 1 ]; then
  echo "ERROR: server not ready at $URL/health (is ./start_server.sh running?)" >&2
  exit 1
fi
echo "[gens] server ready"

hdr() { grep -i "^$2:" "$1" | awk '{print $2}' | tr -d '\r'; }

# --- Gen 0: FORWARD, new session ---
echo
echo "=== Gen 0: forward_1chunk.npz (new session) ==="
code=$(curl -s -w '%{http_code}' -X POST "$URL/sequence/generate" \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@$IMAGE" \
  -F "trajectory=@$TRAJ/forward_1chunk.npz" \
  -F "resolution=240,416" -F "num_frames=81" -F "prompt=$PROMPT" \
  --max-time 600 -D "$OUT/h0.txt" -o "$OUT/seq_gen0.mp4") \
  || { echo "ERROR: Gen 0 request failed (network/timeout)." >&2; exit 1; }
if [ "$code" != "200" ]; then
  echo "ERROR: Gen 0 returned HTTP $code:" >&2; cat "$OUT/seq_gen0.mp4" >&2; echo >&2; exit 1
fi
SID="$(hdr "$OUT/h0.txt" X-Session-Id)"
echo "[gens] session=$SID  cache_entries=$(hdr "$OUT/h0.txt" X-Cache-Entries)  -> $OUT/seq_gen0.mp4"

# --- Gen 1: ROTATE LEFT, continue the same session (no image) ---
echo
echo "=== Gen 1: rotate_left_1chunk.npz (continue session $SID) ==="
code=$(curl -s -w '%{http_code}' -X POST "$URL/sequence/generate" \
  -H "Authorization: Bearer $TOKEN" \
  -F "session_id=$SID" \
  -F "trajectory=@$TRAJ/rotate_left_1chunk.npz" \
  -F "resolution=240,416" -F "num_frames=81" -F "prompt=$PROMPT" \
  --max-time 600 -D "$OUT/h1.txt" -o "$OUT/seq_gen1.mp4") \
  || { echo "ERROR: Gen 1 request failed (network/timeout)." >&2; exit 1; }
if [ "$code" != "200" ]; then
  echo "ERROR: Gen 1 returned HTTP $code:" >&2; cat "$OUT/seq_gen1.mp4" >&2; echo >&2; exit 1
fi
echo "[gens] cache_entries=$(hdr "$OUT/h1.txt" X-Cache-Entries)  -> $OUT/seq_gen1.mp4"

echo
echo "DONE — spatial memory grew across the two generations (X-Cache-Entries above)."
echo "Outputs:"
ls -la "$OUT"/seq_gen0.mp4 "$OUT"/seq_gen1.mp4

#!/usr/bin/env bash
# recipes/tp1.sh — {{MODEL}} on ONE DGX Spark (TP1), TensorFold. Normally started via ./run.sh tp1.
#
# TODO: one paragraph — why these settings (memory fit, context, concurrency, anything unusual).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"

PARALLEL="${PARALLEL:-8}"; CONTEXT="${CONTEXT:-auto}"
build_all

check_model "$MODEL_DIR"; [ "${DRAFTS:-1}" = 0 ] || check_model "$DRAFT_DIR"
run_container run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR:/models/{{SLUG}}:ro" -v "$CACHE_DIR:/root/.cache" "${SPEC_MOUNT[@]}" \
  "${BASE_ENV[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    tensorfold serve /models/{{SLUG}} "${ENDPOINT_ARGS[@]}" "${SERVE_ARGS[@]}" "${SPEC_ARGS[@]}" ${EXTRA:-}
[ "${DRY_RUN:-0}" = 1 ] || echo "launched $NAME tp=1 parallel=$PARALLEL context=${CONTEXT} drafts=${DRAFTS:-1}"

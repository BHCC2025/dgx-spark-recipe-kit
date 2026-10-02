#!/usr/bin/env bash
# recipes/tp2.sh — {{MODEL}} across TWO DGX Sparks (TP2) over one QSFP cable, TensorFold with two CUDA ranks.
# Normally started via ./run.sh tp2 (worker rank 1 first, then the head, rank 0, which serves the API).
#   recipes/tp2.sh 0|1    run one rank on the current node
#
# The network is NCCL over RoCE on the one cabled CX7 port of each node (cluster.env TP2_*, kit pair profile); rank 1
# meets rank 0 at TP2_HEAD_IP:MPORT. Both ranks get the same PARALLEL / CONTEXT / KV_DTYPE / drafting settings.
# TODO: one paragraph — why these settings (memory fit, context, concurrency, anything unusual).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
RANK="${1:?rank 0 or 1}"

PARALLEL="${PARALLEL:-8}"; CONTEXT="${CONTEXT:-auto}"
MPORT="${MPORT:-29551}"                         # TensorFold's rendezvous port (its default)
build_all

case "$RANK" in
  0) RANK_ARGS=("${ENDPOINT_ARGS[@]}") ;;
  1) RANK_ARGS=() ;;
  *) echo "rank must be 0 or 1" >&2; exit 2 ;;
esac
nccl_env_pair "$RANK"                         # kit/lib/nccl.sh: NCCL over RoCE on the one cabled port

check_model "$MODEL_DIR"; [ "${DRAFTS:-1}" = 0 ] || check_model "$DRAFT_DIR"
run_container run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_DIR:/models/{{SLUG}}:ro" -v "$CACHE_DIR:/root/.cache" "${SPEC_MOUNT[@]}" \
  "${BASE_ENV[@]}" "${NCCL_ENV[@]}" ${DOCKER_EXTRA:-} \
  "$IMAGE" \
    tensorfold serve /models/{{SLUG}} "${RANK_ARGS[@]}" "${SERVE_ARGS[@]}" "${SPEC_ARGS[@]}" \
    --tp 2 --rank "$RANK" --master "$TP2_HEAD_IP" --master-port "$MPORT" ${EXTRA:-}
[ "${DRY_RUN:-0}" = 1 ] || echo "launched $NAME rank=$RANK tp=2 parallel=$PARALLEL context=${CONTEXT} drafts=${DRAFTS:-1}"

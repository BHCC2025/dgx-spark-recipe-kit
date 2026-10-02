# shellcheck shell=bash disable=SC2034  # the arrays built here are used by recipes/tpN.sh
# lib/common.sh — shared by recipes/tpN.sh. Sourced, not run.
# Builds the bash arrays each launcher splices into `docker run ... tensorfold serve`. Every knob is an env var with a
# per-TP default that the launcher sets before calling build_all.

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$REPO_DIR/cluster.env" ] \
  || { echo "no $REPO_DIR/cluster.env — run ./setup.sh (or cp cluster.env.example cluster.env and edit it)" >&2; exit 2; }
# cluster.env, with anything already set in the environment taking precedence (PORT=8001 ./run.sh tp1)
# shellcheck disable=SC1091
source "$REPO_DIR/kit/lib/cluster_env.sh"; load_cluster_env "$REPO_DIR/cluster.env"
# NCCL network profile (pair) lives in the shared kit so ./setup.sh tests exactly these settings.
# shellcheck disable=SC1091
source "$REPO_DIR/kit/lib/nccl.sh"

# The image is built by ./setup.sh from docker/Dockerfile: TensorFold from upstream at a pinned commit, unmodified.
IMAGE="${IMAGE:-{{IMAGE}}}"
NAME="${NAME:-${CONTAINER_NAME:-{{CONTAINER}}}}"
PORT="${PORT:-8000}"
SERVED_NAMES="${SERVED_NAMES:-{{SERVED}}}"
CACHE_DIR="${CACHE_DIR:-/var/tmp/{{SLUG}}-tensorfold-cache}"   # compiled kernels + prompt snapshots, per node
DRAFT_DIR="${DRAFT_DIR:-{{DRAFT_DIR}}}"

# Variables forwarded from the head to the workers so every rank runs the same config.
FORWARD_VARS=(IMAGE NAME PARALLEL CONTEXT KV_DTYPE DRAFTS PORT MPORT MODEL_DIR DRAFT_DIR CACHE_DIR
              IB_GID_INDEX IB_GID_INDEX_TP2 NCCL_DEBUG NCCL_CHANNELS EXTRA DOCKER_EXTRA)
forward_env() {
  local v out=""
  for v in "${FORWARD_VARS[@]}"; do [ -n "${!v+x}" ] && out+="$v=$(printf %q "${!v}") "; done
  printf '%s' "$out"
}

# Start rank N on a worker over SSH. The repo must exist at the same path there (./setup.sh copies it).
start_remote_rank() {  # host rank script
  local host=$1 rank=$2 script=$3
  echo "== rank $rank on $host"
  ssh -o BatchMode=yes "$host" "cd $(printf %q "$REPO_DIR") && $(forward_env) bash $(printf %q "$script") $rank" \
    || { echo "rank $rank on $host failed to start" >&2; exit 1; }
}

stop_on() {  # host...
  local h
  for h in "$@"; do ssh -n -o BatchMode=yes "$h" "docker rm -f $NAME" >/dev/null 2>&1 || true; done
}

# ==== model-specific ==================================================================================================
# TODO: any `tensorfold serve` flags this model needs (most need none: TensorFold reads the family from config.json).
MODEL_ARGS=()

# Drafting. DRAFTS=1 (default) drafts with the draft model in DRAFT_DIR when the recipe has one, otherwise with what the
# family brings (e.g. an MTP head); DRAFTS=0 is TensorFold's serial reference (--no-drafts: same replies, slower).
build_spec() {
  SPEC_ARGS=(); SPEC_MOUNT=()
  case "${DRAFTS:-1}" in
    1) if [ -n "$DRAFT_DIR" ]; then SPEC_ARGS=(--drafter /models/draft); SPEC_MOUNT=(-v "$DRAFT_DIR:/models/draft:ro"); fi ;;
    0) SPEC_ARGS=(--no-drafts) ;;
    *) echo "DRAFTS must be 1 or 0" >&2; exit 2 ;;
  esac
}
# ======================================================================================================================

build_misc() {
  local n
  # first served name is the model id; the rest answer as aliases
  # shellcheck disable=SC2206
  local names=($SERVED_NAMES)
  NAME_ARGS=(--name "${names[0]}"); for n in "${names[@]:1}"; do NAME_ARGS+=(--alias "$n"); done
  CONTEXT_ARGS=(); [ "${CONTEXT:-auto}" != auto ] && CONTEXT_ARGS=(--context "$CONTEXT")
  BASE_ENV=(-e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions)
  # Every rank: the same parallel, context, cache and drafting settings (TensorFold requires them to agree).
  SERVE_ARGS=(--parallel "$PARALLEL" "${CONTEXT_ARGS[@]}" --kv-dtype "${KV_DTYPE:-bf16}" "${MODEL_ARGS[@]}")
  # Rank 0 only: the HTTP endpoint.
  ENDPOINT_ARGS=("${NAME_ARGS[@]}" --host 0.0.0.0 --port "$PORT")
}

build_all() { build_spec; build_misc; }

check_model() {  # dir — DRY_RUN=1 only notes a missing model, so the commands can be printed before any download
  [ -z "$1" ] || [ -f "$1/config.json" ] && return 0
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "# note: no model at $1 on $(hostname) yet — ./setup.sh downloads it" >&2; return 0; fi
  echo "MODEL MISSING at $1 on $(hostname) — run ./setup.sh" >&2; exit 3
}

# DRY_RUN=1 prints the docker command instead of running it.
run_container() {
  if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'docker'; printf ' %q' "$@"; printf '\n'; return 0
  fi
  mkdir -p "$CACHE_DIR"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # Free page cache first: on a unified-memory box it counts against what the GPU can allocate.
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  docker "$@" >/dev/null
  sleep 3
  docker ps --format '{{.Names}} {{.Status}}' | grep "^$NAME " || { echo "$NAME exited"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
}

#!/usr/bin/env bash
# tests/test_scripts.sh — the kit's shell pieces without any hardware: new-recipe.sh for every TP range and both engines
# (no leftover placeholders, DRY_RUN commands for every size, clear errors), load_cluster_env precedence, and the bench guard
# against a stand-in OpenAI-style server. Needs bash, python3 and curl.
set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T=$(mktemp -d); trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$T"' EXIT
fails=0
ok()   { printf 'ok    %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; fails=$((fails + 1)); }
check() { local name=$1; shift; if "$@" >/dev/null 2>&1; then ok "$name"; else fail "$name"; fi; }

# ---- new-recipe.sh ---------------------------------------------------------------------------------------------
declare -A NAMEOF=([1]=Test-Model-DGX-Spark-TP1 [1-2]=Test-Model-DGX-Spark-TP1-TP2 [1-3]=Test-Model-DGX-Spark-TP1-TP3
                   [2-3]=Test-Model-DGX-Spark-TP2-TP3 [3]=Test-Model-DGX-Spark-TP3)
for tp in 1 1-2 1-3 2-3 3; do
  d="$T/r$tp"
  if ! "$KIT/new-recipe.sh" --model Test-Model --hf org/Test-Model-NVFP4 --tp "$tp" --out "$d" > "$T/gen.out" 2>&1; then
    fail "new-recipe --tp $tp: $(tail -1 "$T/gen.out")"; continue; fi
  grep -q "^name: ${NAMEOF[$tp]}\$" "$d/recipe.yaml" && ok "new-recipe --tp $tp: name ${NAMEOF[$tp]}" || fail "new-recipe --tp $tp: name"
  left=$(grep -rn --exclude=LICENSE -E '\{\{[A-Z_]+\}\}' "$d" | head -3)
  [ -z "$left" ] && ok "new-recipe --tp $tp: no placeholders left" || fail "new-recipe --tp $tp: placeholders left: $left"
  python3 - "$d" "$tp" <<'PY' && ok "new-recipe --tp $tp: links resolve, no markers left, text fits the sizes" || fail "new-recipe --tp $tp: docs"
import os, re, sys
d, tp = sys.argv[1], sys.argv[2]
bad = []
for root, _, files in os.walk(d):
    for f in files:
        if not f.endswith(".md"): continue
        p = os.path.join(root, f); s = open(p).read()
        if "<!--" in s: bad.append(f"{f}: marker left")
        for link in re.findall(r"\]\(([^)#:]+)\)", s):
            if not os.path.exists(os.path.join(root, link)): bad.append(f"{f}: broken link {link}")
r = open(os.path.join(d, "README.md")).read()
if tp == "1":
    for w in ("worker", "fabric", "NCCL", "networking.md"):
        if w in r: bad.append(f"README (TP1 only) mentions {w}")
hw = next(l for l in r.splitlines() if l.startswith("| Hardware |"))
if ("DGX Sparks" in hw) != (tp != "1"): bad.append(f"Hardware row grammar: {hw}")
if "3" not in tp and "TP3" in r: bad.append("README mentions TP3")
if "2" not in tp.replace("1-3", "123").replace("2-3", "23") and "TP2:" in r: bad.append("README mentions TP2")
if bad: print("\n".join(bad)); sys.exit(1)
PY
  for f in run.sh lib/common.sh recipes/*.sh bench/bench.sh scripts/smoke-test.sh setup.sh; do
    (cd "$d" && bash -n $f) || fail "new-recipe --tp $tp: bash -n $f"; done
  python3 - "$d/recipe.yaml" <<'PY' && ok "new-recipe --tp $tp: recipe.yaml parses" || fail "new-recipe --tp $tp: recipe.yaml"
import sys
try:
    import yaml
except ImportError:
    sys.exit(0)   # PyYAML not installed here; CI installs it
y = yaml.safe_load(open(sys.argv[1]))
assert [r["id"] for r in y["recipes"]] == [f"tp{n}" for n in y["setup"]["tp_sizes"]]
assert sorted(y["setup"]["first_run"]) == y["setup"]["tp_sizes"]
PY
  # a working recipe: the kit next to it, the example cluster.env, model flags filled in
  ln -s "$KIT" "$d/kit"; cp "$d/cluster.env.example" "$d/cluster.env"
  sed -i.bak 's|^MODEL_ARGS=(TODO)|MODEL_ARGS=(--dtype auto)|' "$d/lib/common.sh"
  for n in 1 2 3; do
    [ -f "$d/recipes/tp$n.sh" ] || continue
    c=$(cd "$d" && DRY_RUN=1 ./run.sh tp$n 2>/dev/null | grep -c '^docker ')
    [ "$c" = "$n" ] && ok "new-recipe --tp $tp: DRY_RUN tp$n prints $n docker command(s)" || fail "new-recipe --tp $tp: DRY_RUN tp$n gave $c"
  done
done
check "new-recipe refuses to overwrite"       bash -c "! '$KIT/new-recipe.sh' --model Test-Model --hf org/x --tp 1-2 --out '$T/r1-2'"
check "new-recipe rejects a bad --tp"          bash -c "! '$KIT/new-recipe.sh' --model X --hf org/x --tp 4 --out '$T/bad'"
check "new-recipe rejects a bad --model"       bash -c "! '$KIT/new-recipe.sh' --model 'a b' --hf org/x --tp 1 --out '$T/bad'"
d="$T/r1-2"
check "run.sh refuses a size the recipe lacks" bash -c "cd '$d' && ! DRY_RUN=1 ./run.sh tp3"
check "real launch without the model stops"    bash -c "cd '$d' && ./run.sh tp1 2>&1 | grep -q 'MODEL MISSING'"

# ---- new-recipe.sh --engine tensorfold --------------------------------------------------------------------------
declare -A TFNAME=([1]=Test-Model-DGX-Spark-TP1-TensorFold [2]=Test-Model-DGX-Spark-TP2-TensorFold
                   [1-2]=Test-Model-DGX-Spark-TP1-TP2-TensorFold)
for tp in 1 2 1-2; do
  d="$T/tf$tp"
  if ! "$KIT/new-recipe.sh" --engine tensorfold --model Test-Model --hf org/Test-Model-MLX-4bit --draft org/Test-Draft \
       --tp "$tp" --out "$d" > "$T/gen.out" 2>&1; then
    fail "tensorfold --tp $tp: $(tail -1 "$T/gen.out")"; continue; fi
  grep -q "^name: ${TFNAME[$tp]}\$" "$d/recipe.yaml" && ok "tensorfold --tp $tp: name ${TFNAME[$tp]}" || fail "tensorfold --tp $tp: name"
  left=$(grep -rn --exclude=LICENSE -E '\{\{[A-Z_]+\}\}' "$d" | head -3)
  [ -z "$left" ] && ok "tensorfold --tp $tp: no placeholders left" || fail "tensorfold --tp $tp: placeholders left: $left"
  [ ! -e "$d/engines" ] && [ -f "$d/docker/Dockerfile" ] && [ ! -e "$d/recipes/tp3.sh" ] \
    && ok "tensorfold --tp $tp: overlay applied (Dockerfile, no engines/, no tp3)" || fail "tensorfold --tp $tp: overlay"
  python3 - "$d" "$tp" <<'PY' && ok "tensorfold --tp $tp: links resolve, no markers left, text fits the sizes" || fail "tensorfold --tp $tp: docs"
import os, re, sys
d, tp = sys.argv[1], sys.argv[2]
bad = []
for root, _, files in os.walk(d):
    for f in files:
        if not f.endswith(".md"): continue
        p = os.path.join(root, f); s = open(p).read()
        if "<!--" in s: bad.append(f"{f}: marker left")
        if "vLLM" in s or "GMU" in s: bad.append(f"{f}: mentions vLLM settings")
        for link in re.findall(r"\]\(([^)#:]+)\)", s):
            if not link.startswith("http") and not os.path.exists(os.path.join(root, link)): bad.append(f"{f}: broken link {link}")
r = open(os.path.join(d, "README.md")).read()
if not r.splitlines()[2].startswith("Built on [TensorFold]"): bad.append("README: no 'Built on TensorFold' opening line")
if tp == "1":
    for w in ("worker", "fabric", "NCCL", "networking.md"):
        if w in r: bad.append(f"README (TP1 only) mentions {w}")
if bad: print("\n".join(bad)); sys.exit(1)
PY
  for f in run.sh lib/common.sh recipes/*.sh bench/bench.sh scripts/smoke-test.sh setup.sh; do
    (cd "$d" && bash -n $f) || fail "tensorfold --tp $tp: bash -n $f"; done
  python3 - "$d/recipe.yaml" "$KIT/lib/recipe.py" <<'PY' && ok "tensorfold --tp $tp: recipe.yaml parses, setup sees build + draft" || fail "tensorfold --tp $tp: recipe.yaml"
import subprocess, sys
try:
    import yaml
except ImportError:
    sys.exit(0)
y = yaml.safe_load(open(sys.argv[1]))
assert y["engine"]["name"] == "tensorfold" and y["engine"]["build"] == "docker/Dockerfile"
assert y["setup"]["draft_dir"] == "/var/tmp/models/Test-Draft" and y["model"]["draft"]["hf_repo"] == "org/Test-Draft"
assert [r["id"] for r in y["recipes"]] == [f"tp{n}" for n in y["setup"]["tp_sizes"]]
out = subprocess.run([sys.executable, sys.argv[2], sys.argv[1]], capture_output=True, text=True, check=True).stdout
assert "R_BUILD=docker/Dockerfile" in out and "R_DRAFT_REPO=org/Test-Draft" in out and "R_ENGINE=tensorfold" in out, out
PY
  ln -s "$KIT" "$d/kit"; cp "$d/cluster.env.example" "$d/cluster.env"
  for n in 1 2; do
    [ -f "$d/recipes/tp$n.sh" ] || continue
    (cd "$d" && DRY_RUN=1 ./run.sh tp$n 2>/dev/null) > "$T/dry.out"
    c=$(grep -c '^docker ' "$T/dry.out")
    [ "$c" = "$n" ] && grep -q -- '--drafter /models/draft' "$T/dry.out" && grep -q 'tensorfold serve /models/test-model' "$T/dry.out" \
      && ok "tensorfold --tp $tp: DRY_RUN tp$n prints $n tensorfold command(s) with the draft model" \
      || fail "tensorfold --tp $tp: DRY_RUN tp$n gave $c: $(head -c 300 "$T/dry.out")"
    if [ "$n" = 2 ]; then
      r1=$(grep '^docker ' "$T/dry.out" | head -1); r0=$(grep '^docker ' "$T/dry.out" | tail -1)
      [[ "$r1" == *"--rank 1 "* && "$r1" != *"--port"* && "$r0" == *"--rank 0 "* && "$r0" == *"--port 8000"* \
         && "$r0" == *"--master 10.10.20.1"* && "$r1" == *"NCCL_IB_HCA=rocep1s0f1"* ]] \
        && ok "tensorfold --tp $tp: TP2 ranks (endpoint on rank 0 only, pair NCCL profile)" || fail "tensorfold --tp $tp: TP2 rank args"
    fi
  done
done
c=$(cd "$T/tf1-2" && DRY_RUN=1 DRAFTS=0 PARALLEL=4 CONTEXT=65536 ./run.sh tp2 2>/dev/null | grep -c -- '--no-drafts --tp 2')
[ "$c" = 2 ] && grep -q . <(cd "$T/tf1-2" && DRY_RUN=1 PARALLEL=4 CONTEXT=65536 ./run.sh tp2 2>/dev/null | grep -- '--parallel 4 --context 65536') \
  && ok "tensorfold knobs reach both ranks (DRAFTS=0, PARALLEL, CONTEXT)" || fail "tensorfold knobs ($c)"
check "new-recipe refuses tensorfold at TP3"      bash -c "! '$KIT/new-recipe.sh' --engine tensorfold --model X --hf org/x --tp 1-3 --out '$T/bad'"
check "new-recipe refuses --draft for vllm"        bash -c "! '$KIT/new-recipe.sh' --model X --hf org/x --draft org/y --tp 1 --out '$T/bad'"

# ---- concurrent-summary.py ------------------------------------------------------------------------------------------
cell='{"prompt": "code", "temperature": 0.0, "streams": 8, "failed": 0, "aggregate_tps": 300.0, "per_stream_tps": 40.0, "ttft_s_max": 0.1, "alone": {"equal": 8, "unequal": %d, "failed": 0}}'
printf '{"cells": [%s]}' "$(printf "$cell" 0)" > "$T/c-ok.json"; printf '{"cells": [%s]}' "$(printf "$cell" 1)" > "$T/c-bad.json"
python3 "$KIT/bench/concurrent-summary.py" "$T/c-ok.json" | grep -q "ALL EQUAL" && ! python3 "$KIT/bench/concurrent-summary.py" "$T/c-bad.json" >/dev/null \
  && ok "concurrent summary: passes equal replies, fails an unequal one" || fail "concurrent summary"

# ---- load_cluster_env --------------------------------------------------------------------------------------------
cat > "$T/c.env" <<'X'
PORT=8000
SERVED_NAMES="a b"   # comment
NODES=(n1 n2)
EMPTY=
X
r=$(env -u PORT bash -c "source '$KIT/lib/cluster_env.sh'; load_cluster_env '$T/c.env'; echo \"\$PORT|\$SERVED_NAMES|\${NODES[*]}\"")
[ "$r" = "8000|a b|n1 n2" ] && ok "load_cluster_env: file values" || fail "load_cluster_env: file values ($r)"
r=$(PORT=8001 SERVED_NAMES="x" NODES=bogus bash -c "source '$KIT/lib/cluster_env.sh'; load_cluster_env '$T/c.env'; echo \"\$PORT|\$SERVED_NAMES|\${NODES[*]}\"")
[ "$r" = "8001|x|n1 n2" ] && ok "load_cluster_env: environment wins, arrays from the file" || fail "load_cluster_env: override ($r)"

# ---- bench guard ---------------------------------------------------------------------------------------------------
cat > "$T/fake.py" <<'PY'
import http.server, json, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = json.dumps({"data": [{"id": sys.argv[2], "max_model_len": 262144}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
    do_POST = lambda self: (self.send_response(500), self.end_headers())
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
python3 "$T/fake.py" 18471 other-model & python3 "$T/fake.py" 18472 test-model &
for _ in $(seq 20); do curl -sf 127.0.0.1:18471/v1/models >/dev/null && curl -sf 127.0.0.1:18472/v1/models >/dev/null && break; sleep 0.2; done
d="$T/r1-2"; before=$(ls "$d/bench/results" | wc -l)
(cd "$d" && WAIT=5 bash bench/bench.sh guard http://127.0.0.1:18471/v1 > "$T/g.out" 2>&1); rc=$?
[ "$rc" = 2 ] && grep -q "Nothing written" "$T/g.out" && [ "$(ls "$d/bench/results" | wc -l)" = "$before" ] \
  && ok "bench refuses another model and writes nothing" || fail "bench guard (rc=$rc: $(head -1 "$T/g.out"))"
(cd "$d" && WAIT=5 timeout 30 bash bench/bench.sh own http://127.0.0.1:18472/v1 > "$T/g2.out" 2>&1)
grep -q "^model=test-model" "$T/g2.out" && compgen -G "$d/bench/results/*-own.log" >/dev/null \
  && ok "bench accepts the recipe's own model" || fail "bench own model: $(head -2 "$T/g2.out" | tr '\n' ' ')"
mkdir -p "$T/adhoc"; (cd "$d" && BENCH_OUT="$T/adhoc" WAIT=5 timeout 30 bash bench/bench.sh x http://127.0.0.1:18471/v1 >/dev/null 2>&1)
compgen -G "$T/adhoc/*-x.log" >/dev/null && ! compgen -G "$d/bench/results/*-x.log" >/dev/null \
  && ok "BENCH_OUT sends another model's results elsewhere" || fail "BENCH_OUT"

echo; [ "$fails" = 0 ] && echo "ALL PASS" || { echo "$fails FAILED"; exit 1; }

# dgx-spark-recipe-kit

The shared setup kit for the BHCC2025 DGX Spark recipe repos. Every recipe carries a copy under `kit/`, so a user
gets the same experience whatever model they're setting up:

```bash
git clone https://github.com/BHCC2025/<recipe>.git && cd <recipe>
./setup.sh            # nodes, SSH, dependencies, cabling, fabric IPs, cluster.env, image, network test, model
./run.sh tpN
```

## Recipes using the kit

| Recipe | Sparks | Engine | Single-stream decode, code (fastest setup) |
|---|---|---|---|
| [Qwen3.8-Flash-Next-DGX-Spark-TP1-TP3](https://github.com/BHCC2025/Qwen3.8-Flash-Next-DGX-Spark-TP1-TP3) | 1–3 | vLLM | 57.4 tok/s (3 Sparks) |
| [Gemma-4-31B-IT-DGX-Spark-TP1-TP2](https://github.com/BHCC2025/Gemma-4-31B-IT-DGX-Spark-TP1-TP2) | 1–2 | vLLM | 40.6 tok/s (2 Sparks) |

Every number is benched on our own Sparks with the recipe's `bench/bench.sh`; see each recipe's `bench/results/`.

## What `setup.sh` does

| Step | What it does | Changes anything? |
|---|---|---|
| 1. Nodes | Asks how many Sparks and their SSH names; checks passwordless SSH | offers `ssh-copy-id` |
| 2. Inspect | Runs `lib/probe.py` on every node: OS, GPU, memory, tools, Docker, disk, LAN, every RDMA port | no |
| 3. Dependencies | Docker without sudo, rsync, perftest, the Hugging Face CLI | offers `apt install`, docker group, a local venv for `hf` |
| 4. Cabling | Works out which port is cabled to which node (IPv6 all-nodes ping, so no IPv4 is needed yet), checks the shape (1 cable / triangle), checks IPs and the RoCE v2 GID | offers to assign fabric IPs (NetworkManager connection or netplan, only on those ports) |
| 5. cluster.env | Picks the model directory (checks free space on every node) and writes the detected values into `cluster.env` | shows a diff, asks |
| 6. Image | Pulls the pinned image on every node and checks the GPU is visible inside it | offers `docker pull` |
| 7. Network test | `ping` and `ib_write_bw` on every cable, then an NCCL all-reduce **inside the recipe's image with the recipe's NCCL settings** | no (runs a short-lived container) |
| 8. Model | `hf download` at the pinned revision on the head, `rsync` to the workers, then the recipe's prepare commands | asks first |

- `--check` changes nothing and downloads nothing. It writes `.setup/report.txt`, which is what to attach to an issue.
- `--yes` accepts every default. You can re-run setup at any time.

## Files

| File | Purpose |
|---|---|
| `setup.sh` | the entry point (a recipe's `./setup.sh` just runs `kit/setup.sh`) |
| `lib/probe.py` | per-node facts as JSON, stdlib only, read-only |
| `lib/topology.py` | probes → cables, validation, IP plan, `cluster.env` values |
| `lib/nccl.sh` | NCCL network profiles `pair` (1 cable) and `triangle` (3 cables). Recipes source this, so the setup test and the real run use identical settings |
| `lib/nccl_check.py` | torch.distributed all-reduce with a correctness check and bus bandwidth |
| `lib/netconfig.sh` | persistent static IP on one CX7 port |
| `lib/recipe.py` | reads `recipe.yaml` for setup |
| `template/` | starting files for a new recipe repo |

## Recipe repo standard

Every recipe repo looks like this:

```
<Model>-DGX-Spark-TP<min>-TP<max>/     name always carries the TP range (search)
  README.md            sections, in order: quick-glance table · requirements · quick start (setup.sh, per node count)
                       · settings · how it works · benchmarks · troubleshooting · credits · license
  recipe.yaml          metadata + the `setup:` block the kit reads (template/recipe.yaml)
  cluster.env.example  every key the recipe reads; setup.sh fills in the detected ones
  setup.sh             exec kit/setup.sh "$@"
  run.sh               tpN | stop | status | logs; DRY_RUN=1 prints the docker commands
  lib/common.sh        shared launcher pieces; sources kit/lib/nccl.sh for the network
  recipes/tpN.sh       one launcher per TP size
  patches/<name>/      each with PROVENANCE.md (source, commit, author, what changed, sha256)
  scripts/             recipe-specific helpers (prepare steps, smoke test)
  bench/               bench.sh + results/<date>-<tp>.md and raw logs
  docs/                how it works, networking notes, troubleshooting
  kit/                 this repo (git subtree)
  LICENSE NOTICE CHANGELOG.md
```

Rules:
- **Every TP size a recipe ships gets benched on our own Sparks** with `bench/bench.sh` before `recipe.yaml` marks
  it `verified: true`. Numbers taken from upstream are labelled as upstream figures until then.
- Pin the image (tag and digest) and the model revision.
- Credit upstream work in NOTICE and PROVENANCE.md. Don't copy code across incompatible licences.
- No hardcoded hostnames, IPs or home paths anywhere outside `cluster.env`.
- Publish private first; flip to public after review.

## Updating the kit inside a recipe repo

Recipes vendor a tagged release, never `main`:

```bash
git subtree pull --prefix kit https://github.com/BHCC2025/dgx-spark-recipe-kit.git vX.Y.Z --squash
```

## License

Apache-2.0.

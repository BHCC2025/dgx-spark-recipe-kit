# Troubleshooting

Start with `./setup.sh --check` (it changes nothing), then look at the server log (`./run.sh logs`{{WORKER_LOGS}}).

| Symptom | Cause | Fix |
|---|---|---|
| Download fails with 401/403 | Gated model: licence not accepted, or not logged in | Accept the licence on the model's Hugging Face page, then `hf auth login` |
| `MODEL MISSING at ...` | The model (or the draft model) isn't on that node | `./setup.sh` (downloads it and copies it to every node) |
| `image missing` / `Unable to find image` | The image is built locally, never pulled | `./setup.sh` builds it from `docker/Dockerfile` on the head and copies it to the workers |
| The first start takes several minutes | TensorFold compiles its CUDA kernels on first use | Wait; they are kept in `CACHE_DIR` (per node), so later starts are quick |
<!-- multi -->
| Rank 0 waits forever for rank 1 | Wrong fabric values in `cluster.env`, a firewall on `MPORT`, or rank 1 exited | `docker logs {{CONTAINER}}` on the worker; `./run.sh stop` and start again |
| A rank exits or hangs right after loading | The ranks were started with different `PARALLEL` / `CONTEXT` / drafting settings (TensorFold requires them to agree) | Start both with `./run.sh`, which forwards every knob to the worker |
| Multi-node run much slower than expected | NCCL fell back to TCP sockets | Start with `NCCL_DEBUG=INFO`; the log should show `NET/IB`. Re-run `./setup.sh --check` |
<!-- /multi -->
| OOM or a Spark reboots while loading | Other containers are holding memory; page cache | Stop everything else, then check `free -g` |
| Context smaller than expected | `CONTEXT=auto` sizes the window to the memory left after `PARALLEL` streams | Lower `PARALLEL`, or set `CONTEXT` explicitly (TensorFold refuses one that cannot fit) |
| A setting from the environment is ignored | It isn't one of the knobs the launchers read | The Settings table in the README lists them; a new knob also needs adding to `FORWARD_VARS` in `lib/common.sh` so the workers get it |
| TODO | model-specific problems (draft model, context length) | |

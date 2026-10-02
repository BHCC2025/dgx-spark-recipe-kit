#!/usr/bin/env python3
"""concurrent-summary.py FILE.json — one line per cell of a TensorFold bench_concurrent.py result (the CONCURRENT test
in bench.sh). Exits non-zero if any concurrent reply differed from the same request alone, or any request failed."""
import json, sys

d = json.load(open(sys.argv[1]))
bad = 0
for c in d["cells"]:
    a = c.get("alone") or {}
    bad += a.get("unequal", 0) + a.get("failed", 0) + c.get("failed", 0)
    print(f"{c['prompt']:<5} T={c['temperature']} {c['streams']} user(s): {c['aggregate_tps']:7.1f} tok/s total, "
          f"{c['per_stream_tps']:6.1f} per user, TTFT max {c['ttft_s_max']} s, "
          f"equal to the request alone {a.get('equal', 0)}/{sum(a.values()) if a else 0}")
print("CONCURRENT " + ("ALL EQUAL" if bad == 0 else f"{bad} UNEQUAL OR FAILED"))
sys.exit(1 if bad else 0)

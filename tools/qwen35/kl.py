"""tools/qwen35/kl.py - teacher-forced KL for Qwen3.6-35B-A3B (NEW-MODELS-PLAN.md T1/T2): on one token sequence per
prompt (the prompt + llama.cpp GPU's greedy answer, from a parity.py run), the next-token distributions of
llama.cpp HIP (the reference P), llama.cpp CPU (the noise floor) and strata-q36ref.  Reported: mean KL(P || Q) and
the top-1 agreement over all steps, per engine; the plan's gate is Strata's KL within 1.5x of the floor.

    python tools/qwen35/kl.py --ref BIN --dump BIN --model GGUF --parity DIR --out DIR [--lock "..."] [--cpu-lock "..."]
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

V = 248320


def ids(path: Path, key: str) -> list[int]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(key + ":"):
            return [int(x) for x in line.split(":", 1)[1].split()]
    raise SystemExit(f"no {key}: in {path}")


def run(cmd: list[str], lock: str, log: Path) -> None:
    r = subprocess.run((shlex.split(lock) if lock else []) + cmd, capture_output=True, text=True, timeout=3600)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed (rc {r.returncode}), see {log}")


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--parity", required=True, help="a parity.py output dir (its llama.cpp logs give the sequences)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock", default="")
    ap.add_argument("--cpu-lock", default="")
    ap.add_argument("--ref-args", default="", help="extra strata-q36ref arguments (e.g. '--act f32')")
    a = ap.parse_args()
    out, par = Path(a.out), Path(a.parity)
    out.mkdir(parents=True, exist_ok=True)
    tags = sorted(p.name.split(".")[0] for p in par.glob("p*.llama.log"))
    kl = {"cpu": [], "strata": []}
    top1 = {"cpu": [], "strata": []}
    for tag in tags:
        prompt, gen = ids(par / f"{tag}.llama.log", "prompt"), ids(par / f"{tag}.llama.log", "gen")
        force = ",".join(map(str, gen))
        toks = ",".join(map(str, prompt))
        f = {k: out / f"{tag}.{k}.full" for k in ("gpu", "cpu", "strata")}
        run([a.dump, "--model", a.model, "--tokens", toks, "--force", force, "--ngl", "99", "--dump-full-logits",
             str(f["gpu"])], a.lock, out / f"{tag}.gpu.log")
        run([a.dump, "--model", a.model, "--tokens", toks, "--force", force, "--ngl", "0", "--dump-full-logits",
             str(f["cpu"])], a.cpu_lock, out / f"{tag}.cpu.log")
        run([a.ref, "--model", a.model, "--tokens", toks, "--force", force, "--dump-full-logits", str(f["strata"]),
             *shlex.split(a.ref_args)], a.lock, out / f"{tag}.strata.log")
        P = log_softmax(np.fromfile(f["gpu"], dtype=np.float32).reshape(-1, V))
        line = f"{tag}: {len(P)} steps"
        for k in ("cpu", "strata"):
            Q = log_softmax(np.fromfile(f[k], dtype=np.float32).reshape(-1, V))
            n = min(len(P), len(Q))
            d = (np.exp(P[:n]) * (P[:n] - Q[:n])).sum(axis=1)
            kl[k].extend(d.tolist())
            top1[k].extend((P[:n].argmax(axis=1) == Q[:n].argmax(axis=1)).tolist())
            line += f"; {k}: KL {d.mean():.5f} top-1 {np.mean(P[:n].argmax(axis=1) == Q[:n].argmax(axis=1)):.3f}"
        for p in f.values():
            p.unlink(missing_ok=True)                  # 250 MB each: the numbers are what is kept
        print(line, flush=True)
    for k in ("cpu", "strata"):
        print(f"{k:6s} vs llama.cpp HIP: mean KL {np.mean(kl[k]):.5f} (p99 {np.percentile(kl[k], 99):.4f}), top-1 "
              f"{np.mean(top1[k]):.4f} over {len(kl[k])} steps")
    print(f"ratio strata / floor: {np.mean(kl['strata']) / max(np.mean(kl['cpu']), 1e-12):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

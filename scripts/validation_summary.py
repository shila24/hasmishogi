"""Sanitize a validate_training.sh result directory, extract key results, decide the gate, and package it.

Usage: python3 scripts/validation_summary.py <results-dir>
Writes <results-dir>/summary.json and summary.md and <results-dir>.tar.gz (sanitized logs included).
Exit code 0 only when the gate passes. Standard library only.
"""
from __future__ import annotations

import getpass
import json
import os
import re
import socket
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Steps whose failure blocks production training (see docs/TRAINING_ja.md).
REQUIRED = {"checkout", "env", "build", "sanitize", "tests-native", "tests-depfree", "parity-scaled",
            "parity-scaled-shogi", "benchmark-hasami", "benchmark-riichi", "benchmark-shogi", "devices",
            "cuda-smoke-alphazero", "cuda-smoke-riichi", "cuda-smoke-shogi", "autoloop-smoke",
            "autoloop-install-dry-run", "process-kill-resume"}
RIICHI_MIN_SPEEDUP = 10.0
SHOGI_MIN_SPEEDUP = 10.0

# Skips permitted under Gate PASS (docs/TRAINING_ja.md section 5); any other skip fails the step.
ALLOWED_SKIPS = {
    "Set HASAMI_GUI_TESTS=1 to run real Tk events": "opt-in real Tk event tests (GUI covered by gui-smoke)",
    "optional mahjong library is not installed": "optional scoring oracle; its dependency approval is on hold",
    "demo checkpoint not present": "gitignored demo weights are not part of a checkout",
    "Apple MPS device is not available": "the MPS league collection parity case runs only on Apple GPUs",
    "Set HASAMI_GUI_TESTS=1 for real Tk events": "opt-in real Tk event tests of the shogi GUI (covered by gui-smoke-shogi)",
    "Set SHOGI_SLOW=1 for start perft(4)": "opt-in slow Python perft(4); native perft(5) runs in the ctests",
}
DEPENDENCY_FREE_SKIPS = {
    "torch is optional", "torch not installed", "numpy is optional", "numpy not installed",
    "torch/numpy not installed (use .venv)", "calibrated advisor loading requires torch, numpy and jsonschema",
    "Actual shogi adapter requires numpy",
}


def _secrets() -> list[tuple[str, str]]:
    pairs = [(str(ROOT), "<repo>"), (str(Path.home()), "~")]
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no user database entry is fine
        user = ""
    host = socket.gethostname()
    for name, label in ((host, "<host>"), (host.split(".")[0], "<host>"), (user, "<user>")):
        if name and len(name) >= 3:
            pairs.append((name, label))
    return sorted(set(pairs), key=lambda pair: -len(pair[0]))


PATTERNS = [
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b"), "<mac>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"(?i)([A-Z]:\\Users\\)[^\\\s]+"), r"\1<user>"),
    (re.compile(r"(/mnt/[a-z]/Users/)[^/\s]+"), r"\1<user>"),
    (re.compile(r"(/home/)[^/\s]+"), r"\1<user>"),
]


def sanitize(text: str, secrets: list[tuple[str, str]]) -> str:
    for value, label in secrets:
        if label in ("<user>", "<host>"):
            text = re.sub(rf"(?<![\w.-]){re.escape(value)}(?![\w-])", label, text)
        else:
            text = text.replace(value, label)
    for pattern, replacement in PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _json_tail(text: str):
    """The last top-level JSON object in a log (scripts print one JSON document at the end)."""
    for start in [m.start() for m in re.finditer(r"(?m)^\{", text)][::-1]:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            continue
    return None


def _skips(text: str, step: str) -> tuple[dict, dict]:
    """Skip reasons from verbose unittest output, split into permitted and not permitted."""
    reasons: dict[str, int] = {}
    for reason in re.findall(r"skipped '((?:[^'\\]|\\.)*)'", text):
        reasons[reason] = reasons.get(reason, 0) + 1
    allowed = set(ALLOWED_SKIPS) | (DEPENDENCY_FREE_SKIPS if step == "tests-depfree" else set())
    return ({r: n for r, n in reasons.items() if r in allowed}, {r: n for r, n in reasons.items() if r not in allowed})


def _unittest(text: str) -> dict:
    ran = re.findall(r"Ran (\d+) tests? in ([\d.]+)s", text)
    status = re.findall(r"(?m)^(OK(?: \([^)]*\))?|FAILED \([^)]*\))\s*$", text)
    out = {"ran": int(ran[-1][0]) if ran else None, "seconds": float(ran[-1][1]) if ran else None,
           "status": status[-1] if status else None}
    skipped = re.search(r"skipped=(\d+)", out["status"] or "")
    out["skipped"] = int(skipped.group(1)) if skipped else 0
    out["ok"] = bool(out["status"] and out["status"].startswith("OK"))
    return out


def _ctest(text: str) -> dict:
    found = re.findall(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", text)
    warnings = sorted(set(line.strip() for line in text.splitlines() if "warning:" in line))
    out = {"warnings": len(warnings), "warning_lines": warnings[:20]}
    if found:
        out.update(passed_percent=int(found[-1][0]), failed=int(found[-1][1]), total=int(found[-1][2]))
    return out


def _train(text: str) -> dict:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and '"kind"' in line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    collect = [r for r in rows if r.get("kind") == "collect"]
    update = [r for r in rows if r.get("kind") == "update"]
    losses = [r.get("loss") for r in update]
    return {"collect_seconds": [round(r.get("seconds", 0), 3) for r in collect],
            "plies": [r.get("plies") for r in collect],
            "losses": losses,
            "finite_losses": bool(losses) and all(isinstance(x, (int, float)) and x == x and abs(x) != float("inf")
                                                 for x in losses),
            "checkpoint": "final checkpoint:" in text}


def analyse(name: str, text: str, rc: int) -> tuple[dict, bool]:
    """Key results of one step and whether its content checks passed (beyond the exit code)."""
    if name in ("tests-native", "tests-depfree", "parity-scaled", "parity-scaled-shogi"):
        info = _unittest(text)
        info["parity_lines"] = [line.strip() for line in text.splitlines()
                                if re.search(r"(?i)\bparity\b.*\d", line)][:10]
        info["permitted_skips"], info["unpermitted_skips"] = _skips(text, name)
        if info["skipped"] and not (info["permitted_skips"] or info["unpermitted_skips"]):
            info["unpermitted_skips"] = {"skip reasons not listed (run unittest with -v)": info["skipped"]}
        return info, info["ok"] and not info["unpermitted_skips"]
    if name == "checkout":
        return {"lines": text.strip().splitlines()[:4]}, True
    if name in ("build", "sanitize"):
        info = _ctest(text)
        return info, info.get("failed") == 0 and info.get("total", 0) >= 3
    if name == "benchmark-hasami":
        data = _json_tail(text) or {}
        leaves = {r.get("backend"): r.get("leaves") for r in data.get("results", [])}
        rates = {r.get("backend"): round(r.get("leaf_nodes_per_second", 0)) for r in data.get("results", [])}
        return {"leaves": leaves, "leaf_nodes_per_second": rates}, bool(leaves) and set(leaves.values()) == {3717}
    if name == "benchmark-riichi":
        data = _json_tail(text) or {}
        rates = {f"{r.get('backend')}{'+enc' if r.get('encoding') else ''}": round(r.get("steps_per_second", 0))
                 for r in data.get("results", [])}
        speedup = data.get("native_encoding_speedup")
        return ({"steps_per_second": rates, "native_encoding_speedup": speedup},
                isinstance(speedup, (int, float)) and speedup >= RIICHI_MIN_SPEEDUP)
    if name == "benchmark-shogi":
        data = _json_tail(text) or {}
        speedup = data.get("native_bridge_encoding_speedup")
        return ({"median_steps_per_second": data.get("median_steps_per_second"),
                 "native_bridge_encoding_speedup": speedup},
                isinstance(speedup, (int, float)) and speedup >= SHOGI_MIN_SPEEDUP)
    if name == "devices":
        data = _json_tail(text) or {}
        return data, bool(data.get("cuda_available"))
    if name in ("cuda-smoke-alphazero", "cuda-smoke-riichi", "cuda-smoke-shogi"):
        info = _train(text)
        return info, info["finite_losses"] and info["checkpoint"]
    if name == "autoloop-smoke":
        lines = [line for line in text.strip().splitlines() if line.strip()]
        return {"status": lines[-1] if lines else None}, bool(lines) and lines[-1].startswith("complete")
    if name == "process-kill-resume":
        data = _json_tail(text) or {}
        return data, bool(data.get("ok"))
    if name == "env":
        return _json_tail(text) or {}, True
    if name in ("benchmark-decisions", "benchmark-collection", "benchmark-cuda-sweep"):
        return {"result": _json_tail(text)}, True
    return {"tail": text.strip().splitlines()[-5:]}, True


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    results = Path(sys.argv[1]).resolve()
    secrets = _secrets()
    for path in sorted(results.rglob("*")):
        if path.is_file() and path.suffix in (".log", ".json", ".tsv", ".txt"):
            path.write_text(sanitize(path.read_text(encoding="utf-8", errors="replace"), secrets), encoding="utf-8")
    steps = []
    table = results / "steps.tsv"
    for line in (table.read_text().splitlines() if table.is_file() else []):
        name, status, rc, seconds = line.split("\t")
        log = results / "logs" / f"{name}.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        info, content_ok = analyse(name, text, int(rc)) if status == "ran" else ({}, False)
        passed = status == "ran" and int(rc) == 0 and content_ok
        entry = {"step": name, "required": name in REQUIRED, "status": status, "exit_code": int(rc),
                 "seconds": float(seconds), "passed": passed, "details": info}
        if not passed and text:
            entry["log_tail"] = text.strip().splitlines()[-25:]
        steps.append(entry)
    seen = {s["step"] for s in steps}
    missing = sorted(REQUIRED - seen)
    failed = [s["step"] for s in steps if s["required"] and not s["passed"]]
    gate = "PASS" if not failed and not missing else "FAIL"
    summary = {"gate": gate, "failed_required": failed, "missing_required": missing, "steps": steps}
    (results / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = [f"# Linux/WSL2 validation summary", "", f"**Gate: {gate}**", ""]
    if failed or missing:
        lines += [f"- failed required steps: {', '.join(failed) or '-'}",
                  f"- missing required steps: {', '.join(missing) or '-'}", ""]
    lines += ["| Step | Required | Result | Seconds | Key results |", "|---|---|---|---|---|"]
    for s in steps:
        key = json.dumps({k: v for k, v in s["details"].items() if k not in ("warning_lines", "tail")},
                         ensure_ascii=False)[:300]
        result = "PASS" if s["passed"] else ("SKIPPED" if s["status"] != "ran" else "FAIL")
        lines.append(f"| {s['step']} | {'yes' if s['required'] else 'no'} | {result} | {s['seconds']:.0f} | `{key}` |")
    lines += ["", "process-kill-resume simulates a process kill (SIGKILL); it is not a power-loss test.",
              "Production league training may start only when the gate is PASS.", ""]
    (results / "summary.md").write_text("\n".join(lines))
    archive = results.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(results, arcname=results.name)
    print("\n".join(lines))
    print(f"archive: {archive.name} (sanitized; review before sharing)")
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

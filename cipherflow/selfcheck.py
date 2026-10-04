"""Machine self-check: does this machine compute the OpenFHE kernels correctly?

    python -m cipherflow selfcheck                  # about 15 min: 3 fresh processes + race check
    python -m cipherflow selfcheck --quick          # about 5 min
    python -m cipherflow selfcheck --memtest 8G     # also a memtester pass (apt install memtester)
    python -m cipherflow selfcheck --workload 2     # also the BERT operators in sequence (~10 min each)
    python -m cipherflow selfcheck --mode precise   # the same with two-pass bootstrapping
    python3 cipherflow/selfcheck.py ...             # the same, standard library only

The kernels (native/openfhe/he_ops) are deterministic, so the same input must give the same,
correct result every time. The check runs them in fresh processes (each generates its own keys):

  keys       every rotation key, relinearisation and one bootstrap, against plaintext
  bootstrap  independent inputs, each bootstrapped twice: correct, and the two results bit-identical
  race       (first process) the operations the kernels run in parallel threads, against a serial
             run, bit for bit; Chebyshev evaluation and bootstrapping repeated serially
  workload   (optional) the ten BERT-base operators in MicroBench order in one process, each checked
             against plaintext, then two bootstraps: a fault that the kernels leave behind (in keys
             or bootstrapping tables) shows at the first bootstrap after them
  memtest    (optional) memtester over locked memory

A deterministic computation that gives a wrong or a different result, processes that disagree, or
a memtester failure mean the machine corrupts data (usually memory): results computed on it cannot
be trusted, and timings measured on it are still fine. A failure that every process shares is a
build or library problem instead. A pass does not rule out rare faults; run more processes
(--processes 12) and a memory test for a thorough check.

Writes a JSON report (--out). Exit status: 0 pass, 1 a fault was found, 2 the check could not run.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
_NATIVE = _REPO / "native" / "openfhe"
HE_OPS_GB = 23              # peak memory of one he_ops process (N = 2^16 with all its keys)


# ---------------------------------------------------------------- machine

def _meminfo() -> Dict[str, int]:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0])                  # kB
    except OSError:
        pass
    return out


def _ecc() -> Dict:
    mcs = sorted(Path("/sys/devices/system/edac/mc").glob("mc[0-9]*"))
    if not mcs:
        return {"status": "unknown",
                "note": "no EDAC memory controller registered (ECC off or unsupported, or the edac driver "
                        "is not loaded); as root: dmidecode -t memory | grep -i 'error correction'"}
    def total(name):
        n = 0
        for m in mcs:
            try:
                n += int((m / name).read_text())
            except (OSError, ValueError):
                pass
        return n
    return {"status": "reported", "controllers": len(mcs),
            "corrected_errors": total("ce_count"), "uncorrected_errors": total("ue_count")}


def machine_info() -> Dict:
    mem = _meminfo()
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                   if l.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    commit = ""
    try:
        commit = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {"host": socket.gethostname(), "os": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "mem_total_gb": round(mem.get("MemTotal", 0) / 2**20, 1),
            "mem_available_gb": round(mem.get("MemAvailable", 0) / 2**20, 1),
            "hardware_corrupted_kb": mem.get("HardwareCorrupted"),
            "load_avg": [round(x, 1) for x in os.getloadavg()], "ecc": _ecc(), "cipherflow_commit": commit}


# ---------------------------------------------------------------- he_ops

def he_ops_path(path: Optional[str] = None) -> Path:
    return Path(path or os.environ.get("CIPHERFLOW_OPENFHE_OPS", _NATIVE / "build" / "he_ops"))


def build_he_ops() -> Path:
    exe = he_ops_path()
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake not found; build native/openfhe by hand (needs OpenFHE 1.4)")
    print(f"[selfcheck] building {exe} (OpenFHE 1.4 must be installed) ...", flush=True)
    for cmd in (["cmake", "-S", str(_NATIVE), "-B", str(_NATIVE / "build"), "-DCMAKE_BUILD_TYPE=Release"],
                ["cmake", "--build", str(_NATIVE / "build"), "--target", "he_ops", "-j", str(os.cpu_count() or 4)]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"build failed: {' '.join(cmd)}\n{(r.stdout + r.stderr)[-1500:]}")
    return exe


def run_process(exe: Path, mode: str, specs: List[str], tag: str, timeout: float) -> Dict:
    """One fresh he_ops process; prints a line per check as it arrives."""
    t0 = time.time()
    err_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"cipherflow_selfcheck_{os.getpid()}_{tag}.err"
    rows: List[Dict] = []
    with open(err_path, "w") as err:
        p = subprocess.Popen([str(exe), mode] + specs, stdout=subprocess.PIPE, stderr=err, text=True)
        watchdog = threading.Timer(timeout, p.kill)
        watchdog.start()
        try:
            for line in p.stdout:
                if line.startswith("{"):
                    rows.append(json.loads(line))
                    print(f"[selfcheck] {tag}: {_describe(rows[-1])}", flush=True)
            p.wait()
        except KeyboardInterrupt:
            p.kill()
            raise
        finally:
            watchdog.cancel()
    stderr = err_path.read_text()
    err_path.unlink(missing_ok=True)
    setup = next((l for l in stderr.splitlines() if l.startswith("setup")), "")
    return {"tag": tag, "specs": specs, "returncode": p.returncode, "seconds": round(time.time() - t0, 1),
            "setup": setup, "stderr_tail": stderr.strip()[-600:] if p.returncode else "", "rows": rows}


# the workload: MicroBench order; max |error| against plaintext each operator must stay within
WORKLOAD = ["qk:heads=12", "av:heads=12", "ffn1:H=768,F=3072", "ffn2:F=3072,H=768", "gelu:F=3072,degree=27",
            "layernorm:H=768,degree=27", "out_proj", "add:H=768", "softmax:heads=12", "qkv:H=768"]
TOLERANCE = {"GeLU": 1e-2, "LayerNorm": 0.1, "Softmax": 0.05}       # approximations; the rest is exact
EXACT_TOL = 1e-6


def _op_ok(r: Dict) -> bool:
    return 0 <= r["max_abs_err"] <= TOLERANCE.get(r["op"], EXACT_TOL)


def _describe(r: Dict) -> str:
    if "op" in r:
        e = r["max_abs_err"]
        return (f"{r['op']}: " + ("undecodable" if e < 0 else f"error {e:.2g}") + (" ok" if _op_ok(r) else " WRONG")
                + f" ({r['ms'] / 1e3:.1f} s)")
    c = r.get("check")
    if c == "keys":
        return (f"keys {r['keys'] - r['failed']}/{r['keys']} correct"
                + (f" (failed: {r['which']})" if r["failed"] else ""))
    if c == "bootstrap":
        ok = ["ok" if 0 <= e < 1e-2 else ("undecodable" if e < 0 else f"error {e:.2g}") for e in r["err"]]
        return f"bootstrap input {r['i']}: {ok[0]} / {ok[1]}, " + ("bit-identical" if r["identical"] else "DIFFERENT")
    return f"race {c}: {r['mismatches']} of {r['runs']} differ"


# ---------------------------------------------------------------- memtester

def run_memtest(size: str) -> Dict:
    exe = shutil.which("memtester") or ("/usr/sbin/memtester" if Path("/usr/sbin/memtester").exists() else None)
    if exe is None:
        return {"ran": False, "note": "memtester not installed (apt install memtester)"}
    print(f"[selfcheck] memtester {size} 1 (this can take a long time) ...", flush=True)
    t0 = time.time()
    r = subprocess.run([exe, size, "1"], capture_output=True, text=True)
    text = re.sub(r"[\b\r]", "\n", r.stdout + r.stderr)
    failures = [l.strip() for l in text.splitlines() if "FAILURE" in l]
    return {"ran": True, "size": size, "seconds": round(time.time() - t0, 1), "returncode": r.returncode,
            "locked": "locked." in text and "unlocked" not in text, "failures": failures[:20],
            "failure_count": len(failures), "tests_ok": len(re.findall(r"\bok\b", text))}


# ---------------------------------------------------------------- verdict

def verdict(procs: List[Dict], memtest: Optional[Dict]) -> Dict:
    hardware, deterministic, undetermined, incomplete = [], [], [], []
    groups: Dict[str, tuple] = {}              # kind -> (failed processes, passed processes)
    n_boot = n_boot_bad = n_cmp = n_cmp_bad = n_ops = n_ops_bad = 0
    for pr in procs:
        tag, bad = pr["tag"], False
        kind = "workload" if any("op" in r for r in pr["rows"]) else "check"
        bad_procs, good_procs = groups.setdefault(kind, ([], []))
        if pr["returncode"] != 0:
            why = ("killed (SIGKILL; most likely out of memory)" if pr["returncode"] == -9
                   else f"exit status {pr['returncode']}: {pr['stderr_tail'][-200:]}")
            incomplete.append(f"{tag}: {why}")
        for r in pr["rows"]:
            c = r.get("check")
            if "op" in r:
                n_ops += 1
                if not _op_ok(r):
                    n_ops_bad += 1
                    bad = True
                    if not undetermined or not undetermined[-1].startswith(f"{tag}:"):
                        undetermined.append(f"{tag}: {r['op']} wrong (error {r['max_abs_err']:.3g}); later "
                                            "operators in the process follow")
            elif c == "keys" and r["failed"]:
                bad = True
            elif c == "bootstrap":
                n_boot += 2
                wrong = [e for e in r["err"] if not 0 <= e < 1e-2]
                n_boot_bad += len(wrong)
                bad |= bool(wrong)
                if not r["identical"]:
                    hardware.append(f"{tag}: input {r['i']} bootstrapped twice gave different results "
                                    f"(errors {r['err']}); bootstrapping is deterministic")
            elif "runs" in r:
                n_cmp += r["runs"]
                n_cmp_bad += r["mismatches"]
                if r["mismatches"]:
                    where = "in parallel threads (a race or a memory fault)" if r["outer_parallel"] \
                        else "in serial repeats (a memory fault)"
                    hardware.append(
                        f"{tag}: {c} differs from its reference in {r['mismatches']} of {r['runs']} runs {where}")
        (bad_procs if bad else good_procs).append(tag)
    for kind, (bad_procs, good_procs) in groups.items():
        if bad_procs and good_procs:
            hardware.append(f"{kind} processes running the same code disagree: {', '.join(bad_procs)} failed, "
                            f"{', '.join(good_procs)} passed")
        elif len(bad_procs) == 1 and not hardware:
            undetermined.append(f"{bad_procs[0]} gave wrong results; run more processes to tell a random fault "
                                "(some processes pass) from a software, build or library problem (all fail)")
        elif bad_procs and not hardware:
            deterministic.append(f"every {kind} process failed ({', '.join(bad_procs)}): a software, build or "
                                 "library problem (check the OpenFHE version), not a random fault")
    if hardware:                               # explained by the machine: per-process notes are redundant
        undetermined = [u for u in undetermined if "run more processes" not in u]
    ran_memtest = bool(memtest and memtest.get("ran"))
    if ran_memtest and memtest["failure_count"]:
        hardware.append(f"memtester: {memtest['failure_count']} failures, e.g. {memtest['failures'][0]}")
    elif ran_memtest and memtest.get("returncode"):
        incomplete.append(f"memtester exit status {memtest['returncode']} without failures (could not allocate?)")
    if hardware:
        status = "FAIL: this machine corrupts data (memory or hardware fault suspected)"
    elif deterministic:
        status = "FAIL: deterministic failure (software, build or library)"
    elif undetermined:
        status = "FAIL: wrong results, cause not determined"
    elif incomplete or not (procs or ran_memtest):
        status = "INCOMPLETE"
    else:
        status = "PASS"
    return {"status": status, "hardware": hardware, "deterministic": deterministic,
            "undetermined": undetermined, "incomplete": incomplete,
            "bootstraps": n_boot, "bootstraps_wrong": n_boot_bad, "bit_exact_comparisons": n_cmp,
            "comparisons_differing": n_cmp_bad, "workload_operators": n_ops, "workload_operators_wrong": n_ops_bad}


# ---------------------------------------------------------------- driver

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cipherflow selfcheck", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fast", "precise"], default="fast", help="bootstrapping mode (default fast)")
    ap.add_argument("--processes", type=int, default=3, help="fresh he_ops processes (default 3)")
    ap.add_argument("--bootstraps", type=int, default=4, help="inputs per process, each bootstrapped twice (default 4)")
    ap.add_argument("--race-rounds", type=int, default=3, help="rounds of the race check, 0 to skip (default 3)")
    ap.add_argument("--workload", type=int, default=0, metavar="N",
                    help="also run the BERT operators in sequence in N processes (about 10 min each)")
    ap.add_argument("--memtest", default=None, metavar="SIZE", help="also run memtester over SIZE (e.g. 8G)")
    ap.add_argument("--memtest-only", action="store_true", help="only run memtester (no OpenFHE needed)")
    ap.add_argument("--quick", action="store_true", help="1 process, 2 inputs, 1 race round")
    ap.add_argument("--he-ops", default=None, help="he_ops binary (default native/openfhe/build/he_ops)")
    ap.add_argument("--timeout", type=float, default=4 * 3600, help="seconds per process")
    ap.add_argument("--force", action="store_true", help=f"run even with less than {HE_OPS_GB} GB of free memory")
    ap.add_argument("--out", default=None, help="JSON report (default cipherflow_selfcheck_<host>_<time>.json)")
    a = ap.parse_args(argv)
    if a.quick:
        a.processes, a.bootstraps, a.race_rounds = 1, 2, 1
    if a.memtest_only and not a.memtest:
        ap.error("--memtest-only needs --memtest SIZE")

    info = machine_info()
    print(f"[selfcheck] {info['host']}: {info['cpu']}, {info['logical_cpus']} threads, "
          f"{info['mem_total_gb']} GB ({info['mem_available_gb']} GB free), load {info['load_avg']}, "
          f"ECC {info['ecc']['status']}", flush=True)
    if info["ecc"]["status"] == "reported":
        print(f"[selfcheck] ECC counters: {info['ecc']['corrected_errors']} corrected, "
              f"{info['ecc']['uncorrected_errors']} uncorrected", flush=True)

    procs: List[Dict] = []
    if not a.memtest_only:
        exe = he_ops_path(a.he_ops)
        try:
            if not exe.exists():
                exe = build_he_ops()
        except RuntimeError as exc:
            print(f"[selfcheck] cannot run the OpenFHE checks: {exc}", file=sys.stderr)
            return 2
        for k in range(a.processes):
            free = _meminfo().get("MemAvailable", 0) / 2**20
            if free < HE_OPS_GB and not a.force:
                print(f"[selfcheck] stopping: {free:.0f} GB free, one he_ops process needs about {HE_OPS_GB} GB "
                      "(--force to run anyway)", file=sys.stderr)
                break
            specs = ["keycheck", f"bootcheck:n={a.bootstraps}"]
            if k == 0 and a.race_rounds > 0:
                specs.append(f"racecheck:rounds={a.race_rounds}")
            print(f"[selfcheck] process {k + 1}/{a.processes}: he_ops {a.mode} {' '.join(specs)}", flush=True)
            procs.append(run_process(exe, a.mode, specs, f"process {k + 1}", a.timeout))
        for k in range(a.workload):
            free = _meminfo().get("MemAvailable", 0) / 2**20
            if free < HE_OPS_GB and not a.force:
                print(f"[selfcheck] stopping: {free:.0f} GB free (--force to run anyway)", file=sys.stderr)
                break
            specs = WORKLOAD + ["bootcheck:n=2"]
            print(f"[selfcheck] workload {k + 1}/{a.workload}: he_ops {a.mode} (10 operators, then 2 bootstraps)",
                  flush=True)
            procs.append(run_process(exe, a.mode, specs, f"workload {k + 1}", a.timeout))
    memtest = run_memtest(a.memtest) if a.memtest else None

    v = verdict(procs, memtest)
    report = {"machine": info, "mode": a.mode, "processes": procs, "memtest": memtest, "verdict": v,
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = Path(a.out or f"cipherflow_selfcheck_{info['host']}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.write_text(json.dumps(report, indent=1))

    print(f"\n[selfcheck] {v['status']}")
    if procs:
        print(f"  bootstraps: {v['bootstraps'] - v['bootstraps_wrong']}/{v['bootstraps']} correct; "
              f"bit-exact comparisons: {v['bit_exact_comparisons'] - v['comparisons_differing']}/"
              f"{v['bit_exact_comparisons']} identical"
              + (f"; workload operators: {v['workload_operators'] - v['workload_operators_wrong']}/"
                 f"{v['workload_operators']} correct" if v["workload_operators"] else ""))
    if memtest:
        print("  memtester: " + (memtest.get("note", "") if not memtest.get("ran") else
                                 f"{memtest['failure_count']} failures over {memtest['size']}"
                                 + ("" if memtest["locked"] else " (memory not locked: raise ulimit -l)")))
    for line in v["hardware"] + v["deterministic"] + v["undetermined"] + v["incomplete"]:
        print(f"  - {line}")
    if v["status"] == "PASS":
        print("  no fault seen; rare faults are not ruled out (more --processes and --memtest for a thorough check)")
    print(f"  report: {out}")
    return 0 if v["status"] == "PASS" else (2 if v["status"] == "INCOMPLETE" else 1)


if __name__ == "__main__":
    sys.exit(main())

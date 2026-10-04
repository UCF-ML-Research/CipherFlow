"""Machine self-check: the verdict from he_ops self-check rows (no OpenFHE run needed)."""
from cipherflow import selfcheck as sc

KEYS_OK = {"check": "keys", "keys": 84, "failed": 0, "which": ""}
KEYS_BAD = {"check": "keys", "keys": 84, "failed": 1, "which": "bootstrap"}
RACE_OK = {"check": "qk_task", "outer_parallel": True, "runs": 96, "mismatches": 0}


def boot(i, e1=6e-4, e2=6e-4, same=True):
    return {"check": "bootstrap", "i": i, "err": [e1, e2], "identical": same}


def proc(tag, *rows, rc=0):
    return {"tag": tag, "returncode": rc, "rows": list(rows), "stderr_tail": ""}


def test_pass():
    v = sc.verdict([proc("p1", KEYS_OK, boot(0), boot(1), RACE_OK), proc("p2", KEYS_OK, boot(0))], None)
    assert v["status"] == "PASS" and v["bootstraps"] == 6 and v["bit_exact_comparisons"] == 96


def test_same_input_different_result_is_a_hardware_fault():
    v = sc.verdict([proc("p1", KEYS_OK, boot(0), boot(1, 6e-4, -1, same=False))], None)
    assert v["status"].startswith("FAIL: this machine") and "bootstrapped twice" in v["hardware"][0]


def test_processes_that_disagree_point_at_the_machine():
    v = sc.verdict([proc("p1", KEYS_BAD, boot(0, -1, -1)), proc("p2", KEYS_OK, boot(0))], None)
    assert v["status"].startswith("FAIL: this machine") and "disagree" in v["hardware"][0]


def test_every_process_failing_alike_is_a_build_problem():
    v = sc.verdict([proc("p1", KEYS_BAD, boot(0, -1, -1)), proc("p2", KEYS_BAD, boot(0, -1, -1))], None)
    assert v["status"] == "FAIL: deterministic failure (software, build or library)"
    one = sc.verdict([proc("p1", KEYS_BAD, boot(0, -1, -1))], None)
    assert one["status"] == "FAIL: wrong results, cause not determined"


def test_race_mismatch_memtest_failure_and_killed_process():
    race = dict(RACE_OK, mismatches=2)
    assert sc.verdict([proc("p1", KEYS_OK, race)], None)["hardware"]
    mt = {"ran": True, "failure_count": 3, "failures": ["FAILURE: 0xfe != 0xff at offset 0x10."]}
    assert sc.verdict([], mt)["status"].startswith("FAIL: this machine")
    assert sc.verdict([], {"ran": True, "failure_count": 0, "failures": [], "returncode": 0})["status"] == "PASS"
    killed = sc.verdict([proc("p1", KEYS_OK, rc=-9)], None)
    assert killed["status"] == "INCOMPLETE" and "out of memory" in killed["incomplete"][0]


def test_cli_forwards_to_selfcheck(capsys):
    from cipherflow.cli import main
    try:
        main(["selfcheck", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "--memtest" in capsys.readouterr().out


def op(name, err):
    return {"op": name, "max_abs_err": err, "ms": 1000.0}


WORK_OK = [op("Attention_QK_MatMul", 5e-9), op("LayerNorm", 2e-2), op("Softmax", 9e-4), op("GeLU", 6e-4)]


def test_workload_pass_and_tolerances():
    v = sc.verdict([proc("w1", *WORK_OK, boot(0)), proc("w2", *WORK_OK, boot(0))], None)
    assert v["status"] == "PASS" and v["workload_operators"] == 8 and v["workload_operators_wrong"] == 0
    assert not sc._op_ok(op("Out_Projection", 1e-3)) and sc._op_ok(op("LayerNorm", 5e-2))


def test_workload_failing_in_one_process_points_at_the_machine():
    bad = [op("Attention_QK_MatMul", 5e-9), op("LayerNorm", -1), op("Softmax", -1)]
    v = sc.verdict([proc("w1", *bad, boot(0, -1, -1)), proc("w2", *WORK_OK, boot(0))], None)
    assert v["status"].startswith("FAIL: this machine") and v["workload_operators_wrong"] == 2
    assert any("workload processes" in h and "disagree" in h for h in v["hardware"])


def test_workload_failing_everywhere_is_software():
    bad = [op("Attention_QK_MatMul", 5e-9), op("LayerNorm", -1)]
    v = sc.verdict([proc("w1", *bad), proc("w2", *bad), proc("p1", KEYS_OK, boot(0))], None)
    assert v["status"] == "FAIL: deterministic failure (software, build or library)"

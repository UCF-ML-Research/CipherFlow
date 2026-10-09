"""Network measurement between the two parties (cipherflow netbench), and planning with it."""
import json

import pytest

import cipherflow as cf
from cipherflow import netbench
from cipherflow.cli import main


@pytest.fixture(scope="module")
def server():
    srv = netbench.Server("127.0.0.1", 0).start()
    yield f"127.0.0.1:{srv.port}"
    srv.close()


def test_measure_on_loopback(server):
    m = netbench.measure(server, seconds=0.2, rounds=5)
    assert 0 < m.rtt_ms <= m.rtt_p90_ms
    assert m.up_bps > 1e6 and m.down_bps > 1e6
    n = m.network()
    assert n == cf.Network.parse(m.as_option())
    assert n.bandwidth_bps <= min(m.up_bps, m.down_bps) * 1.001
    assert "loopback" in m.summary()
    two = netbench.measure(server, seconds=0.2, rounds=3, streams=2)
    assert two.streams == 2 and two.up_bps > 1e6


def test_no_server_is_an_explicit_error():
    with pytest.raises(ConnectionError, match="netbench --serve"):
        netbench.measure("127.0.0.1:1", timeout=2)
    assert main(["netbench", "--peer", "127.0.0.1:1", "--seconds", "0.1"]) == 2


def test_cli_plans_with_the_measured_network(server, tmp_path, capsys):
    assert main(["netbench", "--ip", server, "--seconds", "0.2", "--rounds", "5"]) == 0
    assert "planning with --network" in capsys.readouterr().out
    out = tmp_path / "plan.json"
    assert main(["compile", "--model", "bert", "--blocks", "1", "--he", "openfhe-fast", "--mpc", "sci",
                 "--ip", server, "--net-seconds", "0.2", "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "network measured to 127.0.0.1" in text
    plan = json.loads(out.read_text())
    m = plan["deployment"]["network_measured"]
    assert m["peer"] == server
    option = netbench.Measurement(**m).as_option()
    assert plan["deployment"]["bandwidth_bps"] == cf.Network.parse(option).bandwidth_bps
    assert main(["compile", "--model", "bert", "--blocks", "1", "--he", "openfhe-fast", "--mpc", "sci",
                 "--network", option, "-o", str(tmp_path / "same.json")]) == 0
    same = json.loads((tmp_path / "same.json").read_text())
    assert same["cost"] == plan["cost"]

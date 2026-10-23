"""Network measurement between the two parties (cipherflow netbench), and planning with it."""
import json
import socket
import threading

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


def test_the_server_stops_after_one_measurement():
    seen = []
    srv = netbench.Server("127.0.0.1", 0, once=True, on_measured=seen.append)
    t = threading.Thread(target=srv.serve, daemon=True)
    t.start()
    netbench.measure(f"127.0.0.1:{srv.port}", seconds=0.1, rounds=3)
    t.join(timeout=10)
    assert not t.is_alive()
    assert seen == srv.clients == ["127.0.0.1"]
    with pytest.raises(ConnectionError, match="netbench --serve"):
        netbench.measure(f"127.0.0.1:{srv.port}", seconds=0.1, rounds=2, timeout=2)


def test_a_kept_server_serves_several_measurements():
    srv = netbench.Server("127.0.0.1", 0).start()
    try:
        for _ in range(2):
            netbench.measure(f"127.0.0.1:{srv.port}", seconds=0.1, rounds=2)
        assert srv.clients == ["127.0.0.1", "127.0.0.1"]
    finally:
        srv.close()


def test_cli_server_stops_when_the_client_is_done(capsys):
    with socket.socket() as s:                       # a free port
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    rc = []
    t = threading.Thread(target=lambda: rc.append(main(["netbench", "--serve", "--host", "127.0.0.1",
                                                         "--port", str(port)])), daemon=True)
    t.start()
    for _ in range(50):                              # wait for the server to listen
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            threading.Event().wait(0.1)
    assert main(["netbench", "--ip", f"127.0.0.1:{port}", "--seconds", "0.1", "--rounds", "3"]) == 0
    t.join(timeout=10)
    assert not t.is_alive() and rc == [0]
    out = capsys.readouterr().out
    assert "stops after one measurement" in out and "measured by 127.0.0.1" in out and "server stopped" in out

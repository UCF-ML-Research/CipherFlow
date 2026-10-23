"""Measure the network between the two parties for planning, instead of giving it by hand.

The server side waits for a measurement; the client measures the round-trip time (ping-pong of one
byte, median over `rounds`) and the bandwidth in each direction (bulk TCP transfer for `seconds`,
timed at the receiver), over TCP like the MPC backends, then tells the server it is done. A server
started with `once=True` (the command line's default) stops then:

    cipherflow netbench --serve                      # on the server (another terminal or machine)
    cipherflow netbench --peer SERVER_IP             # on the client: just measure
    cipherflow compile --torch demo/transformer.py:bert --peer SERVER_IP --steps     # measure, then plan

A plan uses one bandwidth for both directions (conversions send HE -> MPC traffic to the client and
MPC -> HE traffic to the server); `Measurement.network()` takes the slower direction.
Measurements on one host (127.0.0.1) give loopback speeds, not those of a deployment.
"""
from __future__ import annotations

import socket
import socketserver
import statistics
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple

from .deploy import Network

DEFAULT_PORT = 7787
_REQ = struct.Struct("!cd")          # op (P ping, U client->server, D server->client, Q done), seconds
_UP_REPLY = struct.Struct("!qdd")    # bytes received, first and last arrival (server clock)
_CHUNK = 1 << 20
_MAX_SECONDS = 30.0


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise ConnectionError("connection closed during a measurement")
        buf += part
    return bytes(buf)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            op, seconds = _REQ.unpack(_recv_exact(sock, _REQ.size))
        except (ConnectionError, OSError):             # a port probe, or a client that went away
            return
        seconds = min(max(seconds, 0.0), _MAX_SECONDS)
        if op == b"P":                                 # echo one byte at a time until the client closes
            while True:
                b = sock.recv(1)
                if not b:
                    return
                sock.sendall(b)
        elif op == b"U":                               # receive until the client stops sending
            buf, total, first, last = bytearray(_CHUNK), 0, None, None
            view = memoryview(buf)
            while True:
                n = sock.recv_into(view)
                if not n:
                    break
                last = time.monotonic()
                first = first if first is not None else last
                total += n
            sock.sendall(_UP_REPLY.pack(total, first or 0.0, last or 0.0))
        elif op == b"D":                               # send for `seconds`, then close
            data, end = bytes(_CHUNK), time.monotonic() + seconds
            while time.monotonic() < end:
                sock.sendall(data)
            sock.shutdown(socket.SHUT_WR)
        elif op == b"Q":                               # the client has finished a measurement
            self.server.measured(self.client_address[0])
            sock.sendall(b"K")                         # recorded: the client can return


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    once = False
    on_measured: Optional[Callable[[str], None]] = None

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]                        # one line instead of a traceback
        print(f"netbench: the connection from {client_address[0]} failed ({type(exc).__name__}: {exc})",
              file=sys.stderr, flush=True)

    def measured(self, client: str) -> None:
        self.clients.append(client)
        if self.on_measured is not None:
            self.on_measured(client)
        if self.once:                                  # shutdown() waits for serve_forever: not from its thread
            threading.Thread(target=self.shutdown, daemon=True).start()


class Server:
    """The server side. `serve()` blocks; `Server(port=0).start()` serves in the background (tests).

    With `once=True` it stops after one complete measurement, when the client says it is done;
    otherwise it serves until `close()` (or Ctrl-C). `on_measured(client)` is called after each one."""

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT, *, once: bool = False,
                 on_measured: Optional[Callable[[str], None]] = None):
        self._srv = _Server((host, port), _Handler)
        self._srv.once, self._srv.on_measured, self._srv.clients = once, on_measured, []
        self.host, self.port = self._srv.server_address[:2]
        self._thread: Optional[threading.Thread] = None

    @property
    def clients(self) -> List[str]:
        """The clients that completed a measurement, in order."""
        return list(self._srv.clients)

    def start(self) -> "Server":
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve(self) -> None:
        try:
            self._srv.serve_forever()
        finally:
            self._srv.server_close()

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()


@dataclass
class Measurement:
    peer: str
    rtt_ms: float              # median round trip
    rtt_p90_ms: float
    up_bps: float              # client -> server
    down_bps: float            # server -> client
    streams: int
    seconds: float

    def network(self) -> Network:
        """What a plan uses: the median RTT and the slower direction's bandwidth (as as_option()
        rounds them, so `--network <as_option()>` plans the same)."""
        return Network.parse(self.as_option())

    def as_option(self) -> str:
        """The network as a --network value, e.g. '941.2Mbps/0.214ms'."""
        return f"{_num(min(self.up_bps, self.down_bps) / 1e6, 1)}Mbps/{_num(max(self.rtt_ms, 1e-4), 4)}ms"

    def summary(self) -> str:
        loop = " (loopback: one host)" if self.peer.rsplit(":", 1)[0] in ("127.0.0.1", "localhost", "::1") else ""
        asym = max(self.up_bps, self.down_bps) / max(min(self.up_bps, self.down_bps), 1.0)
        lines = [f"network to {self.peer}{loop}: RTT {self.rtt_ms:.3g} ms (p90 {self.rtt_p90_ms:.3g} ms), "
                 f"client->server {_bw(self.up_bps)}, server->client {_bw(self.down_bps)} "
                 f"({self.streams} TCP stream{'s' if self.streams > 1 else ''}, {self.seconds:g} s each way)",
                 f"planning with --network {self.as_option()}"]
        if asym > 2:
            lines.append(f"note: the link is asymmetric ({asym:.1f}x); plans use the slower direction")
        return "\n".join(lines)

    def to_dict(self):
        return asdict(self)


def _num(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".")


def _bw(bps: float) -> str:
    return f"{bps / 1e9:.3g} Gbps" if bps >= 1e9 else f"{bps / 1e6:.3g} Mbps"


def _split(peer: str, port: int) -> Tuple[str, int]:
    host, sep, p = peer.rpartition(":")
    if sep and p.isdigit() and "]" not in p:
        return host.strip("[]"), int(p)
    return peer.strip("[]"), port


def _connect(host: str, port: int, op: bytes, seconds: float, timeout: float):
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        raise ConnectionError(f"no CipherFlow netbench server at {host}:{port} ({exc}); start one there with "
                              f"`cipherflow netbench --serve --port {port}`") from exc
    sock.settimeout(timeout + seconds)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.sendall(_REQ.pack(op, seconds))
    return sock


def _rtts(host, port, rounds, timeout) -> List[float]:
    with _connect(host, port, b"P", 0.0, timeout) as s:
        out = []
        for _ in range(rounds + 2):
            t = time.perf_counter()
            s.sendall(b"x")
            _recv_exact(s, 1)
            out.append((time.perf_counter() - t) * 1e3)
        return out[2:]                                  # the first round trips include connection warm-up


def _up(host, port, seconds, timeout):
    with _connect(host, port, b"U", seconds, timeout) as s:
        data, end = bytes(_CHUNK), time.monotonic() + seconds
        while time.monotonic() < end:
            s.sendall(data)
        s.shutdown(socket.SHUT_WR)
        return _UP_REPLY.unpack(_recv_exact(s, _UP_REPLY.size))


def _down(host, port, seconds, timeout):
    with _connect(host, port, b"D", seconds, timeout) as s:
        buf = bytearray(_CHUNK)
        view, total, first, last = memoryview(buf), 0, None, None
        while True:
            n = s.recv_into(view)
            if not n:
                break
            last = time.monotonic()
            first = first if first is not None else last
            total += n
        return total, first or 0.0, last or 0.0


def _parallel(fn, streams, *args):
    res: List = [None] * streams
    errs: List[BaseException] = []

    def run(i):
        try:
            res[i] = fn(*args)
        except BaseException as exc:                   # noqa: BLE001 -- re-raised below
            errs.append(exc)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise errs[0]
    total = sum(r[0] for r in res)
    span = max(r[2] for r in res) - min(r[1] for r in res)
    if total == 0 or span <= 0:
        raise ConnectionError("the bandwidth measurement received no data")
    return 8.0 * total / span


def _done(host: str, port: int, timeout: float) -> None:
    """Tell the server the measurement is complete, and wait until it has recorded it (best effort:
    older servers close the connection without an answer)."""
    try:
        with _connect(host, port, b"Q", 0.0, timeout) as s:
            s.recv(1)
    except (ConnectionError, OSError):
        pass


def measure(peer: str, *, port: int = DEFAULT_PORT, rounds: int = 50, seconds: float = 2.0,
            streams: int = 1, timeout: float = 10.0) -> Measurement:
    """The client side: measure RTT and both directions' bandwidth to a `cipherflow netbench --serve`,
    then tell the server it is done (a server started with `once=True` stops)."""
    host, port = _split(peer, port)
    rtts = sorted(_rtts(host, port, rounds, timeout))
    up = _parallel(_up, streams, host, port, seconds, timeout)
    down = _parallel(_down, streams, host, port, seconds, timeout)
    _done(host, port, timeout)
    return Measurement(peer=f"{host}:{port}", rtt_ms=statistics.median(rtts),
                       rtt_p90_ms=rtts[min(len(rtts) - 1, int(0.9 * len(rtts)))],
                       up_bps=up, down_bps=down, streams=streams, seconds=seconds)

"""Deployment description: target hardware + network."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

from compiler.state_expanded.cost_model import NetworkSetting

_BW_UNITS = {"kbps": 1e3, "mbps": 1e6, "gbps": 1e9, "bps": 1.0}
_T_UNITS = {"us": 1e-3, "ms": 1.0, "s": 1e3}


def parse_bandwidth(text: Union[str, float, int]) -> float:
    """'10Mbps' -> 1e7 (bits per second). Numbers are taken as bps."""
    if isinstance(text, (int, float)):
        return float(text)
    m = re.fullmatch(r"\s*([\d.]+)\s*([kKmMgG]?bps)\s*", text)
    if not m:
        raise ValueError(f"cannot parse bandwidth {text!r}; use e.g. '100Mbps' or '1Gbps'")
    return float(m.group(1)) * _BW_UNITS[m.group(2).lower()]


def parse_time_ms(text: Union[str, float, int]) -> float:
    """'20ms' -> 20.0. Numbers are taken as milliseconds."""
    if isinstance(text, (int, float)):
        return float(text)
    m = re.fullmatch(r"\s*([\d.]+)\s*(us|ms|s)\s*", text)
    if not m:
        raise ValueError(f"cannot parse time {text!r}; use e.g. '20ms'")
    return float(m.group(1)) * _T_UNITS[m.group(2)]


@dataclass(frozen=True)
class Network:
    bandwidth_bps: float
    rtt_ms: float

    @classmethod
    def parse(cls, text: str) -> "Network":
        """'10Mbps/20ms' -> Network(1e7, 20.0)."""
        bw, _, rtt = text.partition("/")
        return cls(parse_bandwidth(bw), parse_time_ms(rtt or "1ms"))

    def to_setting(self) -> NetworkSetting:
        return NetworkSetting(self.bandwidth_bps, self.rtt_ms)

    def label(self) -> str:
        return self.to_setting().label()


@dataclass(frozen=True)
class Hardware:
    """Target hardware. `profile` names a built-in profile, a profile path, or "auto".
    `local=True` means the deployment target is this machine, so it can be measured."""
    device: str                 # "cpu" | "cuda"
    name: str = ""
    profile: str = ""
    local: bool = False


@dataclass(frozen=True)
class Deployment:
    hardware: Hardware
    network: Network

    @classmethod
    def of(cls, hardware: Union[str, Hardware], network: Union[str, Network]) -> "Deployment":
        """Deployment.of("gpu", "10Mbps/1ms") using a built-in hardware profile name."""
        hw = hardware if isinstance(hardware, Hardware) else Hardware(
            device="cuda" if hardware.startswith("gpu") else "cpu", name=hardware, profile=hardware)
        net = network if isinstance(network, Network) else Network.parse(network)
        return cls(hw, net)

    @classmethod
    def local(cls, network: Union[str, Network], device: str = "") -> "Deployment":
        """Deploy on this machine; with profile="auto", missing costs are measured here."""
        from .measure import local_device
        dev = device or local_device()
        net = network if isinstance(network, Network) else Network.parse(network)
        return cls(Hardware(device=dev, name=f"local-{dev}", profile="auto", local=True), net)

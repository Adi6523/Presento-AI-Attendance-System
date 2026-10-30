r"""
Presento Phase 2B: Raspberry Pi Camera Stream Discovery & Network Resolution.
Provides robust mDNS/hostname discovery, dynamic IP resolution, and connection health probing
to ensure seamless, zero-manual-configuration connectivity between Windows and the Pi.
"""
from __future__ import annotations

import argparse
import ipaddress
import logging
import socket
import time
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger("presento.edge.discovery")

DEFAULT_MDNS_HOSTNAME = "presento-pi.local"
DEFAULT_NETBIOS_HOSTNAME = "presento-pi"
DEFAULT_STREAM_PORT = 5001
KNOWN_FALLBACK_IPS: List[str] = []


def is_ip_address(host: str) -> bool:
    """Check whether a given string is a valid IPv4/IPv6 literal."""
    try:
        ipaddress.ip_address(host.strip())
        return True
    except ValueError:
        return False


def build_candidate_hosts(
    configured_host: Optional[str] = None,
    include_fallbacks: bool = True,
) -> List[str]:
    """
    Build ordered, deduplicated candidate host list for connecting to the Raspberry Pi.

    Priority Order:
    1. Explicitly configured host (if provided and non-empty)
    2. 'presento-pi.local' (mDNS link-local resolution on local subnet)
    3. 'presento-pi' (LAN DNS / NetBIOS / Tailscale MagicDNS resolution)
    4. Fallback candidate hostnames if enabled
    """
    candidates: List[str] = []

    def _add(h: Optional[str]):
        if not h:
            return
        clean = h.strip()
        if clean and clean not in candidates:
            candidates.append(clean)

    # 1. Configured host takes primary priority
    if configured_host:
        _add(configured_host)

    # 2. mDNS hostname
    _add(DEFAULT_MDNS_HOSTNAME)

    # 3. Standard short hostname / MagicDNS
    _add(DEFAULT_NETBIOS_HOSTNAME)

    # 4. Fallback known historical IPs
    if include_fallbacks:
        for fb_ip in KNOWN_FALLBACK_IPS:
            _add(fb_ip)

    return candidates


def resolve_hostname(host: str, timeout: float = 2.0) -> Optional[str]:
    """
    Resolve a hostname or IP to an IPv4 string using system DNS/mDNS resolver.
    Returns resolved IPv4 string, or None if unresolvable.
    """
    clean_host = host.strip()
    if is_ip_address(clean_host):
        return clean_host

    try:
        # socket.getaddrinfo handles mDNS (.local) natively on Windows 10/11
        addr_info = socket.getaddrinfo(
            clean_host,
            None,
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        if addr_info:
            return str(addr_info[0][4][0])
    except (socket.gaierror, socket.herror, OSError):
        pass

    return None


def probe_stream(
    host: str,
    port: int = DEFAULT_STREAM_PORT,
    timeout: float = 1.2,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Probe a target host and port to verify if the Presento TCP camera stream is reachable.

    Returns:
        (is_reachable: bool, resolved_ip: Optional[str], error_message: Optional[str])
    """
    resolved_ip = resolve_hostname(host, timeout=timeout)
    if not resolved_ip:
        return False, None, f"Could not resolve hostname '{host}'"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((resolved_ip, port))
        return True, resolved_ip, None
    except (ConnectionRefusedError, TimeoutError, OSError) as exc:
        return False, resolved_ip, str(exc)
    finally:
        try:
            sock.close()
        except Exception:
            pass


def discover_pi_stream(
    configured_host: Optional[str] = None,
    port: int = DEFAULT_STREAM_PORT,
    timeout: float = 1.0,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Discover the active Raspberry Pi camera stream by checking candidates in priority order.

    Returns:
        (active_host, active_ip) if found, or (None, None) if Pi stream is currently unreachable.
    """
    candidates = build_candidate_hosts(configured_host)
    logger.debug("Probing Pi camera stream candidates: %s", candidates)

    for cand in candidates:
        reachable, resolved_ip, _ = probe_stream(cand, port=port, timeout=timeout)
        if reachable and resolved_ip:
            logger.info("Discovered active Raspberry Pi camera stream at %s (%s):%d", cand, resolved_ip, port)
            return cand, resolved_ip

    return None, None


class HostCandidateRotator:
    """
    Stateful candidate rotator with dynamic re-resolution and exponential backoff.
    Used by FrameReceiver to automatically reconnect when network changes or Pi reboots.
    """

    def __init__(
        self,
        primary_host: Optional[str] = None,
        port: int = DEFAULT_STREAM_PORT,
        min_retry_delay: float = 1.5,
        max_retry_delay: float = 5.0,
        backoff_multiplier: float = 1.5,
    ):
        self.primary_host = primary_host
        self.port = port
        self.min_retry_delay = min_retry_delay
        self.max_retry_delay = max_retry_delay
        self.backoff_multiplier = backoff_multiplier

        self.current_delay = min_retry_delay
        self._candidates = build_candidate_hosts(primary_host)
        self._current_index = 0
        self._last_connected_host: Optional[str] = None
        self._last_connected_ip: Optional[str] = None

    def get_current_candidate(self) -> str:
        """Return the current candidate host."""
        if not self._candidates:
            return DEFAULT_MDNS_HOSTNAME
        return self._candidates[self._current_index % len(self._candidates)]

    def record_connection_success(self, host: str, ip: Optional[str] = None) -> None:
        """Record successful connection; resets retry delay and locks in active host."""
        self._last_connected_host = host
        self._last_connected_ip = ip
        self.current_delay = self.min_retry_delay

        # Put successfully connected host at front of candidate rotation for fast reconnection
        if host in self._candidates:
            self._candidates.remove(host)
            self._candidates.insert(0, host)
            self._current_index = 0

    def record_connection_failure(self) -> float:
        """
        Rotate to next candidate on failure and apply exponential backoff.
        Returns the delay seconds to wait before next attempt.
        """
        delay = self.current_delay
        self.current_delay = min(self.max_retry_delay, self.current_delay * self.backoff_multiplier)
        self._current_index = (self._current_index + 1) % max(1, len(self._candidates))
        return delay

    def reset_candidates(self, primary_host: Optional[str] = None) -> None:
        """Rebuild candidate list (e.g. after config change)."""
        self.primary_host = primary_host
        self._candidates = build_candidate_hosts(primary_host)
        self._current_index = 0
        self.current_delay = self.min_retry_delay


def main(argv: Optional[List[str]] = None):
    """CLI diagnostic tool to test Pi discovery and stream availability."""
    parser = argparse.ArgumentParser(description="Presento Raspberry Pi Camera Stream Discovery Utility")
    parser.add_argument("--host", default=None, help=f"Optional explicit host/IP to test (default: mDNS {DEFAULT_MDNS_HOSTNAME})")
    parser.add_argument("--port", type=int, default=DEFAULT_STREAM_PORT, help=f"Stream port (default: {DEFAULT_STREAM_PORT})")
    parser.add_argument("--timeout", type=float, default=1.5, help="Probe timeout in seconds")
    args, _ = parser.parse_known_args(argv)

    print("=" * 70)
    print("      Presento Raspberry Pi Camera Stream Discovery Diagnostic")
    print("=" * 70)

    target_host = args.host
    if not target_host:
        try:
            from .config import settings
            if settings.pi_ip and settings.pi_ip != DEFAULT_MDNS_HOSTNAME:
                target_host = settings.pi_ip
        except Exception:
            pass

    candidates = build_candidate_hosts(target_host)
    print(f"Target candidates to test: {candidates}\n")

    found = False
    for cand in candidates:
        print(f"Testing candidate: {cand:<25}", end="", flush=True)
        resolved = resolve_hostname(cand, timeout=args.timeout)
        if resolved:
            print(f" -> Resolved: {resolved:<16}", end="", flush=True)
            reachable, _, err = probe_stream(cand, port=args.port, timeout=args.timeout)
            if reachable:
                print(f" [ONLINE] (Stream active on port {args.port})")
                found = True
            else:
                print(f" [OFFLINE] ({err or 'Connection refused'})")
        else:
            print(" [UNRESOLVED] (Hostname not found on local network)")

    print("-" * 70)
    if found:
        print("[SUCCESS] Active Presento camera stream detected!")
    else:
        print("[NOTICE] Camera stream not currently active.")
        print("  - Ensure Raspberry Pi is powered on.")
        print("  - Ensure presento-camera.service is active on the Pi.")
        print("  - If on different Wi-Fi networks, connect both devices to Tailscale.")
    print("=" * 70)


if __name__ == "__main__":
    main()

import time
import subprocess
from pathlib import Path

VPN_FLAG_DIR = Path("/tmp/vpn-flags")
VPN_FLAG_DIR.mkdir(exist_ok=True)

RECONNECT_FLAG = VPN_FLAG_DIR / "vpn-reconnect.flag"
DISCONNECT_FLAG = VPN_FLAG_DIR / "vpn-disconnect.flag"

SDR_IPS = ["192.168.1.31", "192.168.1.32", "192.168.1.33", "192.168.1.34"]

def _any_sdr_responds() -> bool:
    for ip in SDR_IPS:
        result = subprocess.run(
            ["curl", "--silent", "--max-time", "1", f"http://{ip}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            return True
    return False

def _handle_can_reconnect(timeout: float):
    if not RECONNECT_FLAG.exists():
        print("[VPN] Sending reconnect signal...")
        RECONNECT_FLAG.touch()
    else:
        print("[VPN] A reconnect is already in progress. Waiting...")

    print("[VPN] Waiting for reconnection to complete...")

    start = time.time()
    while time.time() - start < timeout:
        if (not RECONNECT_FLAG.exists()) and _any_sdr_responds():
            print("[VPN] Reconnection confirmed.")
            return
        time.sleep(0.5)

    raise RuntimeError("VPN reconnection timed out (flag still present or SDRs unresponsive).")

def _handle_can_disconnect(timeout: float):
    if not DISCONNECT_FLAG.exists():
        print("[VPN] Sending disconnect signal...")
        DISCONNECT_FLAG.touch()
    else:
        print("[VPN] A disconnect is already in progress. Waiting...")

    print("[VPN] Waiting for disconnection to complete...")

    start = time.time()
    while time.time() - start < timeout:
        if (not DISCONNECT_FLAG.exists()) and (not _any_sdr_responds()):
            print("[VPN] Disconnection confirmed.")
            return
        time.sleep(0.5)

    raise RuntimeError("VPN disconnection timed out (flag still present or SDRs still responsive).")

def exec_vpn_command(word: str, timeout: float = 10.0):
    if word == "CAN-RECONNECT":
        _handle_can_reconnect(timeout)
        return
    elif word == "CAN-DISCONNECT":
        _handle_can_disconnect(timeout)
        return
    else:
        raise RuntimeError(f"Unsupported VPN command: '{word}'")
    

# ------------------------
# Modular VPN keep-alive
# ------------------------
class VPNKeepAlive:
    """
    Utility to periodically reconnect and disconnect the VPN to avoid long-running issues like rekey failures.

    Needs a VPN reconnector script running as another process that listens for command flags at /tmp/vpn-flags, acting accordingly.
    """
    def __init__(self, reconnect_cmd: str = "CAN-RECONNECT", disconnect_cmd: str = "CAN-DISCONNECT",
                 min_minutes=30, max_minutes=40, timeout=10.0):
        self.reconnect_cmd = reconnect_cmd
        self.disconnect_cmd = disconnect_cmd
        self.timeout = timeout
        self.min_minutes = min_minutes
        self.max_minutes = max_minutes
        self.last_reconnect_time = time.time()
        self.reconnect_interval = self._next_interval()

    def _next_interval(self) -> float:
        """Calculates the next reconnection interval in seconds."""
        from numpy import random as np_random
        return float(60 * (self.min_minutes + (self.max_minutes - self.min_minutes) * np_random.rand()))

    def maybe_reconnect(self, condition=True):
        """ Checks if the VPN should be reconnected based on the condition and time since last reconnection."""
        now = time.time()
        if condition and (now - self.last_reconnect_time > self.reconnect_interval):
            print(f"\n[VPN] Triggering reconnection...")
            exec_vpn_command(self.reconnect_cmd, timeout=self.timeout)
            self.last_reconnect_time = now
            self.reconnect_interval = self._next_interval()
            print(f"[VPN] Next reconnection in {self.reconnect_interval / 60:.2f} minutes.\n")

    def reconnect(self):
        """Triggers a VPN reconnection."""
        print(f"\n[VPN] Triggering reconnection...")
        exec_vpn_command(self.reconnect_cmd, timeout=self.timeout)
        self.last_reconnect_time = time.time()
        self.reconnect_interval = self._next_interval()
        print(f"[VPN] Next reconnection in {self.reconnect_interval / 60:.2f} minutes.\n")

    def disconnect(self):
        """Triggers a VPN disconnection."""
        print(f"\n[VPN] Triggering disconnection...")
        exec_vpn_command(self.disconnect_cmd, timeout=self.timeout)
        print(f"[VPN] Disconnection complete.\n")

# -------------------------
# For direct script usage
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trigger VPN control command.")
    parser.add_argument("word", type=str, help="VPN command (e.g., CAN-RECONNECT, CAN-DISCONNECT)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout in seconds")

    args = parser.parse_args()

    try:
        exec_vpn_command(args.word, timeout=args.timeout)
    except Exception as e:
        print(f"[ERROR] {e}")

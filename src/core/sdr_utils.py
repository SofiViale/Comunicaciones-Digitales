import adi
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

class LoopbackMode(Enum):
    """
    Loopback Modes for the ADALM PLuto SDR.
    They can be:
    - OTA (Over the Air): Loopback Disabled
    - DIGITAL: Signal is internally routed back from Tx to Rx.
    - RF: Not used. "How To Config SDR.ipynb" says that it acts as a repeater.
    """
    OTA:int = 0
    DIGITAL:int = 1
    RF:int = 2

class RxGainControlMode(Enum):
    """
    Gain Control Modes for a SDR receiver. It can be automatic (fast or low attack) or manual.
    """
    MANUAL:str = "manual"
    FAST_ATTACK:str = "fast_attack"
    SLOW_ATTACK:str = "slow_attack"

@dataclass
class RxGainControl:
    """
    Configuration for RX gain control strategy used by the SDR hardware.
    Bundles the mode (RxGainControlMode) and the fixed gain value in dB (only matters when mode is MANUAL).
    """
    mode:RxGainControlMode
    gain: Optional[float]
    
    def __post_init__(self):
        if self.mode == RxGainControlMode.MANUAL:
            if self.gain is None:
                raise ValueError("Gain must be specified in MANUAL mode.")
            if not (MIN_MANUAL_GAIN <= self.gain <= MAX_MANUAL_GAIN):
                raise ValueError(f"Gain {self.gain} dB is out of valid range ({MIN_MANUAL_GAIN} dB, {MAX_MANUAL_GAIN} dB)")
        elif self.gain is not None:
            raise ValueError(f"Gain must be None when mode is {self.mode.name}")

    def to_dict(self):
        """Convert the RxGainControl to a dictionary."""
        return {
            "mode": self.mode.name,
            "gain": self.gain
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'RxGainControl':
        """Create a RxGainControl instance from a dictionary. Accepts the mode as a string or enum."""
        mode = data["mode"]
        if not isinstance(mode, RxGainControlMode):
            mode = RxGainControlMode[mode.strip().upper()]
        gain = data.get("gain")
        return RxGainControl(mode=mode, gain=gain)


MIN_SAMPLE_RATE     = 521_000
MAX_SAMPLE_RATE     = 61_440_000

MIN_RF_BANDWIDTH    = 200_000
MAX_RF_BANDWIDTH    = 20_000_000
SELF_ADJUST_MIN_BW  = 125_000  # If the RF bandwidth is over this value, it will be adjusted to the minimum RF bandwidth.

MIN_LO_FREQ         = 325_000_000
MAX_LO_FREQ         = 3_800_000_000

MIN_MANUAL_GAIN     = 0
MAX_MANUAL_GAIN     = 90

MIN_TX_ATTENUATION  = -90
MAX_TX_ATTENUATION  = 0


PROBLEMATIC_BUFFER_SIZE = 2**20 + 1 # From this size onwards, the Pluto SDR has issues with large buffers in terms of timeouts.
MAX_BUFFER_SIZE = 2**28  # Maximum buffer size for the Pluto SDR. This is a safe upper limit.

@dataclass
class SDRParams:
    # General
    uri: str
    sample_rate: int
    lo_freq: float
    loopback_mode: LoopbackMode
    rf_bandwidth: int

    # Tx
    tx_cyclic_buffer: bool = True
    tx_attenuation: int = -10  # In dB

    # Rx
    rx_gain_control: RxGainControl = RxGainControl(RxGainControlMode.SLOW_ATTACK, None)
    rx_buffer_size: int = 2**20

    def timeout(self) -> Optional[int]:
        return 0 if self.rx_buffer_size >= PROBLEMATIC_BUFFER_SIZE else None
    
    def __post_init__(self):
        if not (MIN_SAMPLE_RATE <= self.sample_rate <= MAX_SAMPLE_RATE):
            raise ValueError(f"Sample rate {self.sample_rate} is out of range ({MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE})")

        self.rf_bandwidth = max(self.rf_bandwidth, MIN_RF_BANDWIDTH)  if self.rf_bandwidth >= SELF_ADJUST_MIN_BW else self.rf_bandwidth
        if not (MIN_RF_BANDWIDTH <= self.rf_bandwidth <= MAX_RF_BANDWIDTH):
            raise ValueError(f"RF bandwidth {self.rf_bandwidth} is out of range ({MIN_RF_BANDWIDTH}, {MAX_RF_BANDWIDTH})")

        if not (MIN_LO_FREQ <= self.lo_freq <= MAX_LO_FREQ):
            raise ValueError(f"LO frequency {self.lo_freq} is out of range ({MIN_LO_FREQ}, {MAX_LO_FREQ})")

        if not (MIN_TX_ATTENUATION <= self.tx_attenuation <= MAX_TX_ATTENUATION):
            raise ValueError(f"Tx attenuation {self.tx_attenuation} dB is out of range ({MIN_TX_ATTENUATION}, {MAX_TX_ATTENUATION})")

        if not (0 < self.rx_buffer_size <= MAX_BUFFER_SIZE):
            raise ValueError(f"Rx buffer size {self.rx_buffer_size} is out of range (0, {MAX_BUFFER_SIZE})")
        
    def to_dict(self):
        return {
            "uri": self.uri,
            "sample_rate": self.sample_rate,
            "lo_freq": self.lo_freq,
            "loopback_mode": self.loopback_mode.name,
            "rf_bandwidth": self.rf_bandwidth,
            "tx_cyclic_buffer": self.tx_cyclic_buffer,
            "tx_attenuation": self.tx_attenuation,
            "rx_gain_control": self.rx_gain_control.to_dict(),
            "rx_buffer_size": self.rx_buffer_size
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'SDRParams':
        return SDRParams(
            uri=data["uri"],
            sample_rate=data["sample_rate"],
            lo_freq=data["lo_freq"],
            loopback_mode=LoopbackMode[data["loopback_mode"]],
            rf_bandwidth=data["rf_bandwidth"],
            tx_cyclic_buffer=data["tx_cyclic_buffer"],
            tx_attenuation=data["tx_attenuation"],
            rx_gain_control=RxGainControl.from_dict(data["rx_gain_control"]),
            rx_buffer_size=data["rx_buffer_size"]
        )


def _log_sdr_info(params: SDRParams):
    def c(text: str, code: int) -> str:
        return f"\033[{code}m{text}\033[0m"

    CYAN    = 96
    GREEN   = 92
    YELLOW  = 93
    RESET   = 0

    rx_mode = params.rx_gain_control.mode.name
    rx_gain = params.rx_gain_control.gain if params.rx_gain_control.mode == RxGainControlMode.MANUAL else "AUTO"
    timeout = params.timeout()

    def fmt_pow2(n: int) -> str:
        from math import log2
        if n & (n - 1) == 0:
            return f"{n} (2^{int(log2(n))})"
        return str(n)

    print(c("\n[INFO] SDR Configuration", GREEN))
    print("=" * 80)

    print(c("\n[Common Settings]", YELLOW))
    print(f"{c('URI'.rjust(25), CYAN)}: {params.uri}")
    print(f"{c('Sample Rate'.rjust(25), CYAN)}: {params.sample_rate} Sps")
    print(f"{c('LO Frequency'.rjust(25), CYAN)}: {params.lo_freq/1e6:.2f} MHz")
    print(f"{c('Loopback Mode'.rjust(25), CYAN)}: {params.loopback_mode.name}")
    print(f"{c('RF Bandwidth'.rjust(25), CYAN)}: {params.rf_bandwidth/1e3:.2f} kHz")

    print(c("\n[Tx Settings]", YELLOW))
    print(f"{c('Cyclic Buffer'.rjust(25), CYAN)}: {params.tx_cyclic_buffer}")
    print(f"{c('Tx Attenuation'.rjust(25), CYAN)}: {params.tx_attenuation} dB")

    print(c("\n[Rx Settings]", YELLOW))
    print(f"{c('Gain Control Mode'.rjust(25), CYAN)}: {rx_mode}")
    print(f"{c('Rx Gain (if manual)'.rjust(25), CYAN)}: {rx_gain}")
    print(f"{c('Rx Buffer Size'.rjust(25), CYAN)}: {fmt_pow2(params.rx_buffer_size)} samples")
    print(f"{c('Rx Timeout'.rjust(25), CYAN)}: {timeout if timeout is not None else 'Default'}")

    print("=" * 80 + "\n")



def init_sdr(params: SDRParams, verbose:bool= False) -> adi.Pluto:
    sdr = adi.Pluto(params.uri)

    # General
    sdr.sample_rate = params.sample_rate
    sdr.loopback = params.loopback_mode.value

    # Tx
    sdr.tx_lo = int(params.lo_freq)
    sdr.tx_rf_bandwidth = int(params.rf_bandwidth)
    sdr.tx_hardwaregain_chan0 = params.tx_attenuation
    sdr.tx_cyclic_buffer = params.tx_cyclic_buffer

    # Rx
    sdr.rx_lo = int(params.lo_freq)
    sdr.rx_rf_bandwidth = int(params.rf_bandwidth)
    sdr.rx_buffer_size = params.rx_buffer_size
    sdr.gain_control_mode_chan0 = params.rx_gain_control.mode.value
    if params.rx_gain_control.mode == RxGainControlMode.MANUAL:
        sdr.rx_hardwaregain_chan0 = params.rx_gain_control.gain

    # Timeout logic for large buffers
    if params.timeout() == 0:
        sdr._ctrl.context.set_timeout(0)
    if verbose:
        _log_sdr_info(params)
    
    return sdr

def change_sdr_attenuation(sdr: adi.Pluto, attenuation: int):
    """
    Change the Tx attenuation of the SDR.
    """
    if not (MIN_TX_ATTENUATION <= attenuation <= MAX_TX_ATTENUATION):
        raise ValueError(f"Attenuation {attenuation} dB is out of range ({MIN_TX_ATTENUATION}, {MAX_TX_ATTENUATION})")
    
    sdr.tx_hardwaregain_chan0 = attenuation


def delete_sdr(sdr:adi.Pluto):
    """
    Delete the SDR object to free up resources.
    """
    if sdr:
        sdr.tx_destroy_buffer()
        sdr.tx_hardwaregain_chan0 = -89 
        sdr.tx_lo                 = int(2400e6)
        sdr.rx_lo                 = int(950e6)
        sdr.tx(np.zeros(2048))
        del sdr
        print("SDR object deleted.")
    else:
        print("No SDR object to delete.")

def soft_delete_sdr(sdr:adi.Pluto, verbose: bool = False):
    """
    Soft delete the SDR object to free up resources.
    """
    if sdr:
        sdr.tx_destroy_buffer()
        del sdr
        if verbose:
            print("[SDR Utils] SDR object soft deleted.")
    else:
        if verbose:
            print("[SDR Utils] No SDR object to delete.")

def optimize_payload_symbols(buffer_len, preamble_symbols, samples_per_symbol, pad_samples=0):
    """
    Calculate the maximum number of payload symbols that can fit in a single frame,
    given a total buffer symbols.

    If `pad_samples > 0`, the function forces the buffer to contain 3 frames
    and selects a payload symbols such that at least one full frame is guaranteed to
    fit completely.

    If `pad_samples == 0`, the buffer only contains 2 frames, but at least one full is guaranteed.

    :param buffer_len: Total length of the buffer in samples.
    :param preamble_symbols: Length of the preamble in symbols.
    :param samples_per_symbol: Number of samples per symbol.
    :param pad_samples: Number of samples to pad each frame with (default is 0, meaning no padding).

    :return: Maximum number of payload symbols that can fit in a single frame.

    :raises ValueError: If the buffer length is too small to fit one complete frame with the given settings.
    """

    # Number of frames assumed in the buffer
    frame_count = 3 if pad_samples > 0 else 2

    # Symbol counts for fixed sections
    sync_word_len = 2
    sfd_len = 2.25
    header_len = 2

    overhead_symbols = preamble_symbols + sync_word_len + sfd_len + header_len
    overhead_samples = int(overhead_symbols * samples_per_symbol + frame_count * pad_samples)

    available_space = buffer_len / frame_count - overhead_samples

    if available_space < samples_per_symbol:
        raise ValueError("Buffer length is too small to fit one complete frame with given settings.")

    max_payload_len = int(available_space / samples_per_symbol)
    return max_payload_len

def remove_iq_drift(
    z: np.ndarray,
    samples_per_symbol: int,
    win_size: int = None,
    hop_size: int = None,
    center_func: Callable = np.mean
) -> np.ndarray:
    """
    Remove IQ drift by estimating a low-rate I/Q centerline using sliding windows
    and subtracting its interpolated value from the original signal.

    Parameters:
    ----------
    z : np.ndarray
        Complex input signal.
    samples_per_symbol : int
        Base unit of symbol size. Default for win/hop size if not specified.
    win_size : int, optional
        Window size used to estimate local IQ centers. Defaults to `samples_per_symbol`.
    hop_size : int, optional
        Hop between window centers. Defaults to `samples_per_symbol`.
    center_func : Callable, optional
        Function used to compute center per window (e.g., np.mean, np.median).

    Returns:
    -------
    np.ndarray
        Drift-corrected complex signal.
    """
    if win_size is None:
        win_size = 2*samples_per_symbol
    if hop_size is None:
        hop_size = samples_per_symbol

    t = np.arange(len(z))
    means_i = []
    means_q = []
    centers = []

    for start in range(0, len(z) - win_size + 1, hop_size):
        end = start + win_size
        segment = z[start:end]
        means_i.append(center_func(segment.real))
        means_q.append(center_func(segment.imag))
        centers.append(start + win_size // 2)

    if len(centers) < 2:
        raise ValueError("Not enough windows to interpolate drift. Try reducing win_size or hop_size.")

    means_i = np.array(means_i)
    means_q = np.array(means_q)
    centers = np.array(centers)

    # Interpolate across the whole signal length
    interp_i = np.interp(t, centers, means_i)
    interp_q = np.interp(t, centers, means_q)
    drift = interp_i + 1j * interp_q

    return z - drift

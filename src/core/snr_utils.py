import numpy as np
import cupy as cp

from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List, Dict

from src.core.sdr_utils import SDRParams
from src.core.params import LoRaPhyParams, LoRaFrameParams

import json
from pathlib import Path
from datetime import datetime

#-------------------------------------------------------
# Generative AWGN utility
#-------------------------------------------------------

def signal_power(signal: np.ndarray) -> float:
    """Return average power after mean removal."""
    return np.mean(np.abs(signal - np.mean(signal))**2)

def generate_awgn(snr_input, full_signal, reference=None):
    """
    Add complex AWGN to full_signal with desired SNR, using the power of `reference`
    (or full_signal if not provided). Compatible with NumPy or CuPy.

    :param snr_input: Desired SNR in dB ({value},"db") or linear scale.
    :param full_signal: Full signal to which noise will be added.
    :param reference: Optional reference signal to compute noise power.
    
    :return: Tuple of (noisy_signal, noise, noise_power).
    :raises ValueError: If SNR is not positive or invalid format.
    """
    if isinstance(snr_input, str) and snr_input.lower().endswith('db'):
        snr_linear = 10 ** (float(snr_input[:-2]) / 10.0)
    else:
        snr_linear = float(snr_input)
    if snr_linear <= 0:
        raise ValueError("SNR must be positive")

    xp = cp.get_array_module(full_signal)  # auto-detect NumPy or CuPy
    ref_signal = reference if reference is not None else full_signal
    sig_var = xp.mean(xp.abs(ref_signal) ** 2)
    noise_var = sig_var / snr_linear

    noise = xp.sqrt(noise_var / 2.0) * (
        xp.random.randn(full_signal.size) + 1j * xp.random.randn(full_signal.size)
    )

    return full_signal + noise, noise, noise_var

#-------------------------------------------------------
# SNR estimation utility
#-------------------------------------------------------

def _coerce_complex_1d(waveform: np.ndarray) -> np.ndarray:
    """Return waveform as a 1-D complex64 NumPy array."""
    return np.asarray(waveform, dtype=np.complex64).ravel()

def _match_lengths(
    received_waveform: np.ndarray,
    ideal_waveform: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncate both arrays to the same (shorter) length.
    
    :param received_waveform: Received waveform array.
    :type received_waveform: np.ndarray
    :param ideal_waveform: Ideal waveform array.
    :type ideal_waveform: np.ndarray

    :return: Tuple of truncated received and ideal waveforms.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    common_length = min(received_waveform.size, ideal_waveform.size)
    return received_waveform[:common_length], ideal_waveform[:common_length]

def _estimate_complex_scale_ls(
    received_waveform: np.ndarray,
    ideal_waveform: np.ndarray,
    *,
    power_floor: float = 1e-18,
) -> complex:
    """
    Compute least-squares complex scale `alpha` mapping ideal -> received.

    `alpha = sum(received * conj(ideal)) / sum(|ideal|**2)`

    :param received_waveform: Measured complex samples (r[k]).
    :type received_waveform: np.ndarray
    :param ideal_waveform: Reference complex samples (s[k]); same length or pre-trimmed.
    :type ideal_waveform: np.ndarray
    :param power_floor: Minimum energy guard; return 0j if ideal energy below this.
    :type power_floor: float

    :return: Complex scale alpha.
    :rtype: complex
    """
    ideal_energy = np.sum(np.abs(ideal_waveform) ** 2)
    if ideal_energy <= power_floor:
        return 0j
    numerator = np.sum(received_waveform * np.conj(ideal_waveform))

    alpha = numerator / ideal_energy
    return alpha

def _snr_from_alpha_and_waveforms(
    received_waveform: np.ndarray,
    ideal_waveform: np.ndarray,
    alpha: complex,
    *,
    power_floor: float = 1e-18,
) -> Tuple[float, float, float]:
    """
    Compute linear SNR plus component powers using a precomputed complex scale.

    signal = alpha * ideal
    noise  = received - signal

    :param received_waveform: Measured complex samples (r[k]).
    :type received_waveform: np.ndarray
    :param ideal_waveform: Reference complex samples (s[k]); same length as received.
    :type ideal_waveform: np.ndarray
    :param alpha: Complex scale mapping ideal -> received.
    :type alpha: complex
    :param power_floor: Minimum noise power guard; if noise <= floor, SNR=inf.
    :type power_floor: float

    :return: (snr_linear, signal_power, noise_power)
    :rtype: Tuple[float, float, float]
    """
    reconstructed_signal = alpha * ideal_waveform
    noise_waveform = received_waveform - reconstructed_signal

    signal_power = np.mean(np.abs(reconstructed_signal) ** 2)
    noise_power = np.mean(np.abs(noise_waveform) ** 2)

    if noise_power <= power_floor:
        return np.inf, signal_power, noise_power

    snr_linear = signal_power / noise_power
    return snr_linear, signal_power, noise_power

def estimate_snr_from_ls_fit(
    received_payload_waveform: np.ndarray,
    ideal_payload_waveform: np.ndarray,
    *,
    return_extras: bool = False,
    power_floor: float = 1e-18,
) -> float | Tuple[float, Dict[str, Any]]:
    """
    LS-fit complex scale `alpha` between aligned received/ideal payloads and return SNR (dB).

    Steps: coerce → length match → `alpha` LS → residual powers → SNR.

    :param received_payload_waveform: Measured payload samples (aligned to payload start).
    :type received_payload_waveform: np.ndarray
    :param ideal_payload_waveform: Remodulated payload samples (reference).
    :type ideal_payload_waveform: np.ndarray
    :param return_extras: If True, also return dict with linear SNR, `alpha`, powers, length.
    :type return_extras: bool
    :param power_floor: Energy/power guard to avoid divide-by-zero.
    :type power_floor: float
    :return: SNR in dB, or (snr_db, extras_dict) if return_extras True.
    :rtype: float | Tuple[float, Dict[str, Any]]
    """
    received = _coerce_complex_1d(received_payload_waveform)
    ideal = _coerce_complex_1d(ideal_payload_waveform)

    if received.size == 0 or ideal.size == 0:
        raise ValueError("Received or ideal waveform is empty.")

    received, ideal = _match_lengths(received, ideal) # LS method requires same length (remember: sum_k)

    alpha = _estimate_complex_scale_ls(
        received, ideal, power_floor=power_floor
    )

    snr_linear, signal_power, noise_power = _snr_from_alpha_and_waveforms(
        received, ideal, alpha, power_floor=power_floor
    )

    snr_db = float("inf") if not np.isfinite(snr_linear) else 10.0 * np.log10(snr_linear)

    if return_extras:
        return snr_db, {
            "snr_linear": snr_linear,
            "alpha": alpha,
            "signal_power": signal_power,
            "noise_power": noise_power,
            "num_samples": ideal.size,
        }

    return snr_db

def estimate_snr_from_ls_fit_segmented(
    received_payload_waveform: np.ndarray,
    ideal_payload_waveform: np.ndarray,
    *,
    sps: int,
    seg_syms: int = 32,
    power_floor: float = 1e-18,
) -> float:
    """
    Divides the received and ideal payload waveforms into segments of `seg_syms` symbols,
    each with `sps` samples per symbol, and estimates SNR for each segment using
    least-squares fitting. Combines the SNRs by summing powers.
    """
    r = np.asarray(received_payload_waveform).ravel()
    s = np.asarray(ideal_payload_waveform).ravel()
    N = min(r.size, s.size)
    r, s = r[:N], s[:N]

    L = seg_syms * sps  # tamaño de segmento en muestras
    sum_Ps = 0.0
    sum_Pn = 0.0

    for i in range(0, N, L):
        rr = r[i:i+L]
        ss = s[i:i+L]
        if rr.size < sps:  # descartar colita demasiado corta
            break

        snr_db_i, ex = estimate_snr_from_ls_fit(
            rr, ss, return_extras=True, power_floor=power_floor
        )
        # Combinar por potencias (ponderado por longitud, usando medias*L)
        Ps_i = ex["signal_power"] * rr.size
        Pn_i = ex["noise_power"]  * rr.size
        sum_Ps += Ps_i
        sum_Pn += Pn_i

    if sum_Pn <= power_floor:
        return float("inf")
    return 10.0 * np.log10(sum_Ps / sum_Pn)


#-------------------------------------------------------
# SDR Profile and SNR utilities
#-------------------------------------------------------

@dataclass
class AttenSNRPoint:
    attenuation: float
    snr_values: List[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.snr_values))

    @property
    def std(self) -> float:
        return float(np.std(self.snr_values))

    def to_dict(self) -> dict:
        """Return a dictionary representation of the attenuation SNR point."""
        return {
            "attenuation": float(self.attenuation),
            "snr_values": [float(snr) for snr in self.snr_values]
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'AttenSNRPoint':
        """Create an AttenSNRPoint instance from a dictionary."""
        return AttenSNRPoint(
            attenuation=data["attenuation"],
            snr_values=data["snr_values"]
        )

from src.core.misc_utils import get_class_persistence_manager

def _slug(s: str) -> str:
    """Keep only [a-zA-Z0-9_-], replace others with '_'."""
    import re
    return re.sub(r'[^0-9A-Za-z_-]+', '_', s).strip('_')

@dataclass
class SDRProfile:
    name: str
    phy_params: LoRaPhyParams
    frame_params: LoRaFrameParams
    fold_mode: str

    tx_sdr_params: Optional[SDRParams] = None
    rx_sdr_params: Optional[SDRParams] = None

    snr_map: List[AttenSNRPoint] = field(default_factory=list)

    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def __post_init__(self):
        sdrs_defined = self.tx_sdr_params is not None and self.rx_sdr_params is not None
        if not sdrs_defined and self.snr_map:
            raise ValueError("snr_map is only valid when both tx_sdr_params and rx_sdr_params are defined.")

    def get_snr_for_atten(self, atten: float) -> float:
        """Interpolate SNR for a given attenuation value using the SNR map."""
        if not self.snr_map:
            raise ValueError("SNR map is empty or undefined.")
        x = [pt.attenuation for pt in self.snr_map]
        y = [pt.mean for pt in self.snr_map]
        return float(np.interp(atten, x, y))

    def get_atten_for_snr(self, target_snr: float) -> float:
        """Interpolate attenuation for a given SNR value using the SNR map."""
        if not self.snr_map:
            raise ValueError("SNR map is empty or undefined.")
        x = [pt.mean for pt in self.snr_map]
        y = [pt.attenuation for pt in self.snr_map]
        return float(np.interp(target_snr, x, y))

    def to_dict(self) -> dict:
        """Convert the SDR profile to a dictionary representation."""
        sdrs_defined = self.tx_sdr_params is not None and self.rx_sdr_params is not None

        return {
            "name": self.name,
            "phy_params": self.phy_params.to_dict(),
            "frame_params": self.frame_params.to_dict(),
            "fold_mode": self.fold_mode,
            "tx_sdr_params": self.tx_sdr_params.to_dict() if self.tx_sdr_params else None,
            "rx_sdr_params": self.rx_sdr_params.to_dict() if self.rx_sdr_params else None,
            "snr_map": [pt.to_dict() for pt in self.snr_map] if sdrs_defined else [],
            "timestamp": self.timestamp
        }
    
    def _uri_to_id(self, uri: Optional[str]) -> str:
        """Take 'ip:192.168.1.34' or 'usb:1.2.3' → '34' or '3' etc. If None ⇒ 'sim'."""
        if not uri:
            return "sim"
        # last chunk after '.' if any (your old logic)
        return uri.split(".")[-1]

    @property
    def device_tag(self) -> str:
        """Tx/Rx device identifier part: '31', '31_32', or 'sim'."""
        tx_id = self._uri_to_id(self.tx_sdr_params.uri) if self.tx_sdr_params else "sim"
        rx_id = self._uri_to_id(self.rx_sdr_params.uri) if self.rx_sdr_params else "sim"
        return tx_id if tx_id == rx_id else f"{tx_id}_{rx_id}"
    
    @property
    def spec_tag(self) -> str:
        """Physical spec tag: 'sf7_bw125k_spc1_FPA'."""
        sf  = self.phy_params.spreading_factor
        bwk = int(self.phy_params.bandwidth / 1e3)
        spc = self.phy_params.samples_per_chip
        return f"sf{sf}_bw{bwk}k_spc{spc}_{self.fold_mode}"

    @property
    def date_tag(self) -> str:
        """Date tag (YYYYMMDD). Uses now, not stored timestamp."""
        return self.timestamp.replace("-", "")  # '2025-07-25' → '20250725'

    def file_stem(self, *, include_name: bool = False) -> str:
        """
        Build the filename stem (no extension).
        Example: 'profile_20250725_31_32_sf7_bw125k_spc1_FPA'
        Optionally include the human 'name' field, slugged.
        """
        parts = ["profile", self.date_tag, self.device_tag, self.spec_tag]
        if include_name and self.name:
            parts.append(_slug(self.name))
        return "_".join(parts)

    def auto_name(self, extension: str = ".json", *, include_name: bool = False) -> str:
        stem = self.file_stem(include_name=include_name)
        ext = extension if extension.startswith(".") else f".{extension}"
        return stem + ext

    def save(self, filename: str | None = None) -> Path:
        pm = get_class_persistence_manager(type(self))
        return pm.save(self, filename=filename, namer=SDRProfile.auto_name)
    
    @classmethod
    def load(cls, filename: str) -> 'SDRProfile':
        pm = get_class_persistence_manager(cls)
        return pm.load(filename, from_dict=cls.from_dict)

    @classmethod
    def list_profiles(cls) -> List[str]:
        """Return available profile names without extension."""
        pm = get_class_persistence_manager(cls)
        return pm.list()

    @staticmethod
    def from_dict(data: dict) -> 'SDRProfile':
        """Create an SDRProfile instance from a dictionary."""
        phy_params = LoRaPhyParams.from_dict(data["phy_params"])
        frame_params = LoRaFrameParams.from_dict(data["frame_params"])

        tx_sdr_params = SDRParams.from_dict(data["tx_sdr_params"]) if data.get("tx_sdr_params") else None
        rx_sdr_params = SDRParams.from_dict(data["rx_sdr_params"]) if data.get("rx_sdr_params") else None

        snr_map_data = data.get("snr_map", [])
        
        if tx_sdr_params and rx_sdr_params:
            snr_map = [AttenSNRPoint.from_dict(pt) for pt in snr_map_data]
        else:
            snr_map = []

        return SDRProfile(
            name=data["name"],
            phy_params=phy_params,
            frame_params=frame_params,
            fold_mode=data["fold_mode"],
            tx_sdr_params=tx_sdr_params,
            rx_sdr_params=rx_sdr_params,
            snr_map=snr_map,
            timestamp=data.get("timestamp", datetime.now().strftime("%Y-%m-%d"))
        )

    def is_simulation(self) -> bool:
        """Check if this profile is for simulation (no SDRs defined)."""
        return self.tx_sdr_params is None and self.rx_sdr_params is None

    def is_single_sdr(self) -> bool:
        """Check if this profile uses a single SDR for both TX and RX."""
        return self.tx_sdr_params is not None and self.rx_sdr_params is not None and \
               self.tx_sdr_params.uri == self.rx_sdr_params.uri
# ------------- Plotting utilities -----------------

def linear_regression(x, y) -> Tuple[float, float]:
    """
    Perform linear regression on x, y data and return slope (m) and intercept (b).
    
    :param x: Independent variable data.
    :param y: Dependent variable data.
    :return: Tuple of slope (m) and intercept (b).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b

def plot_profile_snr_map(profiles: SDRProfile | list[SDRProfile], show_regression: bool = True):
    """
    Plot attenuation vs SNR (mean) for one or multiple SDRProfiles.

    Shows:
    - Scatter points
    - Linear regression line (optional)
    - Interpolated curve
    """
    import matplotlib.pyplot as plt
    if not isinstance(profiles, list):
        profiles = [profiles]

    plt.figure(figsize=(9, 6))
    
    for profile in profiles:
        x = np.array([pt.attenuation for pt in profile.snr_map])
        y = np.array([pt.mean for pt in profile.snr_map])

        # Interpolation
        x_interp = np.linspace(x.min(), x.max(), 300)
        y_interp = np.array([profile.get_snr_for_atten(v) for v in x_interp])

        label_prefix = profile.name

        plt.scatter(x, y, label=f"{label_prefix} (points)")
        plt.plot(x_interp, y_interp, label=f"{label_prefix} (interp)")

        if show_regression:
            m, b = linear_regression(x, y)
            y_fit = m * x + b
            plt.plot(x, y_fit, '--', label=f"{label_prefix} (f: y={m:.2f}x+{b:.2f}) [dB]")

    plt.title("Attenuation vs SNR")
    plt.xlabel("Attenuation [dB]")
    plt.ylabel("SNR [dB]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
LoRa Primitive Helpers
========================

Stateless numerical helpers shared by TX/RX.

All functions take *xp* (array API) as their **first** argument so the caller
can pass either `numpy`, `cupy`, etc.  No global state is mutated;
only a small module-level cache is used for base chirps.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple, List
import numpy as np  # we need pi regardless of backend

from src.core.params import LoRaPhyParams
from src.core.markers import LoRaMarkers

__all__ = [
    "generate_base_chirp",
    "instantaneous_phase",
    "instantaneous_frequency",
    "to_complex",
]


def _as_float(xp, val):
    """Helper: cast *val* to backend float64."""
    return xp.asarray(val, dtype=xp.float64)


# ---------------------------------------------------------------
# 1.  Base-chirp generator with memoised cache
# ---------------------------------------------------------------

_ChirpKey = Tuple[int, int, int, float]  # (SF, SPC, slope, dur)


@lru_cache(maxsize=16)  # keyed by (SF, SPC, slope, dur, backend)
def _cached_chirp(sf: int, spc: int, slope: int, dur: float, backend: str):
    """
    Create (or retrieve) a reference chirp on the requested backend.
    """
    xp = __import__(backend)
    cps = 1 << sf
    sym_len = cps * spc
    n = int(dur * sym_len)

    k = xp.arange(n, dtype=xp.float32)
    coeff = xp.float32(1.0) / xp.sqrt(sym_len, dtype=xp.float32)

    # legacy formula:  φ = ±π k² / (cps · spc²)
    phase = slope * xp.pi * (k**2) / (cps * spc * spc)
    return coeff * xp.exp(1j * phase, dtype=xp.complex64)


def generate_base_chirp(
    xp: Any,
    params: LoRaPhyParams,
    *,
    slope: int = +1,
    duration_factor: float = 1.0,
):
    """Return a normalised base chirp (up or down) as *xp.complex64* array.

    Results are cached **per backend**, so repeated calls are zero-copy.
    """
    if slope not in (+1, -1):
        raise ValueError("slope must be +1 (up) or -1 (down)")
    if duration_factor <= 0:
        raise ValueError("duration_factor must be > 0")

    backend_id = xp.__name__  # e.g. 'numpy', 'cupy'
    return _cached_chirp(
        params.spreading_factor,
        params.samples_per_chip,
        slope,
        duration_factor,
        backend_id,
    )


# ---------------------------------------------------------------
# 2.  Instantaneous phase for arbitrary symbol sequence
# ---------------------------------------------------------------


def instantaneous_phase(
    xp: Any,
    symbol_vals: "list[int] | Tuple[int, ...]",
    params: LoRaPhyParams,
    *,
    slopes: "list[int] | Tuple[int, ...] | None" = None,
) -> Any:
    """Vectorised phase accumulator used by the TX path.

    Parameters
    ----------
    xp : numpy-like module
    symbol_vals : iterable of int 0 ≤ v < 2**SF
    params : ``LoRaPhyParams``
    slopes : iterable of {+1, -1} matching ``symbol_vals`` length.
             If *None*, defaults to *all +1* (standard up-chirps).

    Returns
    -------
    xp.ndarray(dtype=float64)
        Concatenated phase samples.
    """
    cps = params.chips_per_symbol
    spc = params.samples_per_chip
    sym_len = cps * spc

    if slopes is None:
        slopes = (1,) * len(symbol_vals)
    if len(slopes) != len(symbol_vals):
        raise ValueError("slopes and symbol_vals length mismatch")

    phase = xp.empty(len(symbol_vals) * sym_len, dtype=xp.float64)

    F = xp.float64
    pos = 0
    for val, slope in zip(symbol_vals, slopes):
        k = xp.arange(sym_len, dtype=xp.float64)
        k_norm = k / F(sym_len)  # k / (cps*spc)

        term1 = 2 * np.pi * (F(val) + F(slope) * k / (2 * F(spc))) * k_norm
        if slope > 0:
            # Up-chirp wraps back to 0 Hz once it reaches +BW.
            cond = (k / F(spc)) > F(cps - val)
            term2 = -2 * np.pi * ((k / F(spc)) - F(cps - val)) * cond
        else:
            term2 = 0.0
        phase[pos : pos + sym_len] = term1 + term2
        pos += sym_len
    return phase


# ------------------------------------------------------------------
# 3.  Instantaneous frequency  f[k]  for a vector of LoRa symbols
# ------------------------------------------------------------------
def instantaneous_frequency(
    xp: Any,
    symbols: "list[int | LoRaMarkers]",
    phy: LoRaPhyParams,
) -> Any:
    """Return a 1-D array of per-sample frequency [Hz].

    * `symbols` may mix integer payload symbols and ``LoRaMarkers``.
    * Works on NumPy **or** CuPy depending on *xp*.
    """
    cps = phy.chips_per_symbol
    spc = phy.samples_per_chip
    sym_len = cps * spc

    # --- pre-allocate output ------------------------------------------
    total_len = sum(
        int(sym.duration_factor * sym_len) if isinstance(sym, LoRaMarkers) else sym_len
        for sym in symbols
    )
    out = xp.empty(total_len, dtype=xp.float64)

    freq_step = phy.bandwidth / cps
    phase_step = freq_step / spc

    pos = 0
    for sym in symbols:
        if isinstance(sym, int):
            val, slope, dur = sym, +1, 1.0
        else:
            val, slope, dur = sym.symbol_val, sym.slope_sign, sym.duration_factor

        length = int(sym_len * dur)
        k = xp.arange(length, dtype=xp.float64)
        slice_ = val * freq_step + slope * k * phase_step
        if slope > 0:
            slice_ = xp.fmod(slice_, phy.bandwidth)
        out[pos : pos + length] = slice_
        pos += length
    return out


# ---------------------------------------------------------------
# 4.  Phase → complex64 helper
# ---------------------------------------------------------------

def to_complex(xp: Any, phase, coef: float = 1.0):
    """Convert real phase vector → complex baseband samples (complex64)."""
    return (coef * xp.exp(1j * phase)).astype(xp.complex64, copy=False)

# ---------------------------------------------------------------
# 5. Utility to test Phase-Alignment
# ---------------------------------------------------------------
def locate_discontinuity_indices(
    symbols: "list[int | LoRaMarkers]",
    phy: LoRaPhyParams,
) -> Tuple[List[int], List[int]]:
    """
    Finds sample indices of frequency discontinuities in a LoRa symbol stream.

    This function locates discontinuities at symbol boundaries (with specific
    transitions excluded) and intra-symbol phase wraps caused by frequency
    rollovers in data up-chirps.

    :param symbols: The complete sequence of symbols to be analyzed.
    :type symbols: list[int | LoRaMarkers]
    :param phy: The physical layer parameters object.
    :type phy: LoRaPhyParams
    :returns: A tuple containing two lists: the first for boundary
              discontinuity indices, the second for intra-symbol indices.
    :rtype: Tuple[List[int], List[int]]
    """
    boundary_indices = []
    intra_symbol_indices = []
    
    current_sample = 0
    sps = phy.samples_per_symbol
    spc = phy.samples_per_chip
    cps = phy.chips_per_symbol

    # For each symbol in the stream
    for i, sym in enumerate(symbols):
        # Check if there are boundary discontinuities
        if i > 0:
            prev_sym = symbols[i - 1]
            
            # Rule 1: Exclude if symbols are identical, UNLESS they are 0 or a standard up-chirp.
            rule1_exclude = (prev_sym == sym) and (sym not in (0, LoRaMarkers.FULL_UPCHIRP))

            # Rule 2: Exclude the up-chirp to down-chirp transition in the SFD.
            is_prev_up = isinstance(prev_sym, LoRaMarkers) and prev_sym.slope_sign == 1
            is_curr_down = isinstance(sym, LoRaMarkers) and sym.slope_sign == -1
            rule2_exclude = is_prev_up and is_curr_down

            # Rule 3: Exclude if data symbols are "near" each other (uses circular distance).
            rule3_exclude = False
            if isinstance(prev_sym, int) and isinstance(sym, int):
                threshold = 2  # Define what "near" means. Ex: ignore jumps of 1 symbol.
                diff = abs(sym - prev_sym)
                distance = min(diff, cps - diff)

                if distance < threshold:
                    rule3_exclude = True

            if not (rule1_exclude or rule2_exclude or rule3_exclude):
                boundary_indices.append(current_sample)
                
        # Check if there are intra-symbol discontinuities
        if isinstance(sym, (int, np.int64)) and sym > 0 and spc >= 2:
            k_discontinuity = int(spc * (cps - sym)) + 1
            if k_discontinuity < sps:
                intra_symbol_indices.append(current_sample + k_discontinuity)

        samples_in_sym = (
            int(sym.duration_factor * sps) if isinstance(sym, LoRaMarkers) else sps
        )
        current_sample += samples_in_sym

    return boundary_indices, intra_symbol_indices

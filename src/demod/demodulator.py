from __future__ import annotations
from typing import Any, Tuple
from dataclasses import dataclass
from src.core.backend   import choose_backend, AbstractBackend     # xp, fft, plan
from src.core.primitives import generate_base_chirp
from src.core.params    import LoRaPhyParams
from src.core.markers   import LoRaMarkers


K_PHIS = 16  # number of phases for FPA folding

def fold_fpa(xp, fft_, first, last):
    """
    Full Phase Alignment (FPA) folding via brute-force phase search.

    For each symbol, tries K_PHIS different rotations of the second half
    and keeps the one that maximizes the peak magnitude after addition.

    :param xp: NumPy or CuPy module (for CPU/GPU compatibility).
    :type xp: module
    :param fft_: Input FFT array of shape (..., spc * chips).
    :type fft_: xp.ndarray
    :param first: Slice to extract first chip segment.
    :param last: Slice to extract last chip segment.
    :param chips: Number of chips per symbol.

    :return: Phase-aligned folded FFT of shape (..., chips).
    :rtype: xp.ndarray
    """
    a = fft_[..., first]                      # (..., chips)
    b = fft_[..., last]                       # (..., chips)

    phis = xp.linspace(0.0, 2 * xp.pi, K_PHIS, endpoint=False, dtype=a.dtype)  # (K,)
    phasors = xp.exp(1j * phis)               # (K,)

    # Compute all possible combinations
    folded_all = a[..., None, :] + b[..., None, :] * phasors[None, :, None]  # (..., K, chips)

    # Criterion: maximum peak magnitude per candidate
    mags = xp.abs(folded_all)                 # (..., K, chips)
    peak_mags = xp.max(mags, axis=-1)         # (..., K)
    best_idx = xp.argmax(peak_mags, axis=-1)  # (...)

    # Best rotation per symbol
    phi_opt = phis[best_idx]                  # (...)
    folded = a + b * xp.exp(1j * phi_opt[..., None])  # (..., chips)
    return folded


@dataclass(frozen=True, slots=True)
class _FoldSpec:
    """
    Encapsulates oversampling-folding strategies used in FFT-based 
    LoRa demodulation to combine phase-aligned chips.

    Supported modes:
    - '0FPA': Naive folding (no phase alignment).
    - 'FPA':  Fine- Grained Phase Alignment (FPA) folding.
    - 'CPA':  Coarse Phase Alignment (CPA) folding, using absolute values.

    :param mode: Folding mode, one of '0FPA', 'FPA', or 'CPA'.
    :type mode: str
    :param spc: Samples per chip (oversampling factor).
    :type spc: int
    :param chips: Number of chips per symbol.
    :type chips: int
    """
    mode: str
    spc: int
    chips: int

    def build(self, xp):
        """
        Returns a folding function according to the selected strategy.

        :param xp: NumPy or CuPy module (for CPU/GPU compatibility).
        :type xp: module
        :return: A callable that applies the chosen folding method to an FFT array.
        :rtype: Callable[[xp.ndarray], xp.ndarray]
        """
        if self.spc == 1:
            return lambda fft_: fft_[..., :self.chips]

        first = slice(0, self.chips)
        last  = slice((self.spc - 1) * self.chips, self.spc * self.chips)

        if self.mode == "0FPA":
            return lambda fft_: fft_[..., first] + fft_[..., last]

        if self.mode == "FPA":
            return lambda fft_: fold_fpa(xp, fft_, first, last)

        if self.mode == "CPA":
            return lambda fft_: xp.abs(fft_[..., first]) + xp.abs(fft_[..., last])

        raise ValueError(f"Unknown folding mode: {self.mode!r}")


class LoRaDemodulator:
    """
    Unified GPU/CPU LoRa demodulator.

    :param phy_params: LoRaPhyParams
    :param backend: "auto" | "numpy" | "cupy" | AbstractBackend
    :param fold_mode: "0FPA" or "CPA"
    :param safe: If True, move NumPy → GPU and cast to complex64 automatically
    :param return_peaks: If True, also return peak magnitude array
    """

    # ------------------------- construction -----------------------------
    def __init__(self,
                 phy_params: LoRaPhyParams,
                 *,
                 backend: str | AbstractBackend = "auto",
                 fold_mode: str = "0FPA",
                 safe: bool = True
                ):

        self.backend: AbstractBackend = (
            backend if isinstance(backend, AbstractBackend)
            else choose_backend(backend)
        )
        self.xp = self.backend.xp
        self.phy_params = phy_params
        self.safe = safe

        # --- constants --------------------------------------------------
        self.chips   = 1 << phy_params.spreading_factor
        self.sym_len = self.chips * phy_params.samples_per_chip
        self._fold   = _FoldSpec(fold_mode, phy_params.samples_per_chip,
                                 self.chips).build(self.xp)

        # reference chirps on this backend
        self._ref: dict[str, Any] = {}
        self._get_base("downchirp")   # pre-cache


    # ------------------------- public API -------------------------------
    def demodulate(self,
                   buf,
                   *,
                   base: str | LoRaMarkers = "downchirp",
                   return_items: list[str] = ["symbols"]
                   ) -> tuple:
        """
        Demodulate a complex baseband waveform into LoRa symbol indices.

        This function processes a time-domain buffer of I/Q samples and extracts
        the corresponding LoRa symbols. Internally, it performs dechirping, FFT,
        and magnitude folding using the specified backend (NumPy or CuPy).

        :param buf: Input buffer containing I/Q samples, can be a NumPy or CuPy array.
        :type buf: Array-like
        :param base: Reference chirp to use for dechirping, can be a string
                    ("downchirp", "upchirp") or a LoRaMarkers instance.
        :type base: str | LoRaMarkers
        :param return_items: List of items to return, can include:
            - "symbols": Demodulated symbol indices.
            - "peaks": Peak magnitudes of the folded FFT.
            - "folded": Folded FFT magnitudes.
            - "deltas": Delta values for peak detection.
            - "viz_bundle": Visualization bundle with all intermediate results.
        :type return_items: list[str]
        :return: Tuple containing requested items or at least the symbols.
        """
        ALLOWED = {"symbols", "peaks", "folded", "deltas", "viz_bundle"}
        if not set(return_items).issubset(ALLOWED):
            raise ValueError(f"Invalid return_items: {return_items}")

        mat, orig_ndim = self._prepare(buf)               # shape (..., sym, samp)
        dech = self._dechirp(mat, base)                   # elementwise *
        fft  = self.backend.fft(dech, axis=-1)
        folded = self._fold(fft)                            # shape (..., sym, chips)
        mag   = self.xp.abs(folded)                         # shape (..., sym, chips)
        idx   = self.xp.argmax(mag, axis=-1)                # (..., sym)

        symbols_out = self._maybe_squeeze(idx, orig_ndim)

        out = [symbols_out]

        if "peaks" in return_items:
            peaks = self.xp.take_along_axis(mag, idx[..., None], axis=-1).squeeze(-1)
            peaks_out = self._maybe_squeeze(peaks, orig_ndim)
            out.append(peaks_out)
        
        if "folded" in return_items:
            folded_out = self._maybe_squeeze(mag, orig_ndim)
            out.append(folded_out)

        if "deltas" in return_items:
            idx_l = (idx-1) % mag.shape[-1]  # left neighbor
            idx_r = (idx+1) % mag.shape[-1]  # right neighbor
            mag_l = self.xp.take_along_axis(mag, idx_l[..., None], axis=-1).squeeze(-1)
            mag_c = self.xp.take_along_axis(mag, idx[..., None], axis=-1).squeeze(-1)
            mag_r = self.xp.take_along_axis(mag, idx_r[..., None], axis=-1).squeeze(-1)
            denom = (mag_l - 2 * mag_c + mag_r)
            denom_safe = self.xp.where(self.xp.abs(denom) < 1e-12, self.xp.nan, denom)
            deltas = 0.5 * (mag_l - mag_r) / denom_safe
            deltas = self._maybe_squeeze(deltas, orig_ndim)
            out.append(deltas)

        
        if "viz_bundle" in return_items:
            peaks = self.xp.take_along_axis(mag, idx[..., None], axis=-1).squeeze(-1)
            peaks_out = self._maybe_squeeze(peaks, orig_ndim)
            mag_out = mag.reshape(-1, mag.shape[-1])
            debug = {
                "phy_params": self.phy_params,
                "modulated_symbols": buf.get() if hasattr(buf, "get") else buf,
                "demodulated_symbols": symbols_out.get() if hasattr(symbols_out, "get") else symbols_out,
                "peak_magnitudes": peaks_out.get() if hasattr(peaks_out, "get") else peaks_out,
                "folded_mag_fft": mag_out.get() if hasattr(mag_out, "get") else mag_out
            }
            out.append(debug)

        return tuple(out) if len(out) > 1 else out[0]

    # ------------------------- helpers ---------------------------------
    def _prepare(self, buf):
        """
        Cast/reshape to (..., symbols, samples).
        
        This function ensures the input buffer is in the correct format for demodulation.
        It checks the data type, reshapes it to ensure the last dimension matches the symbol length,
        and returns the reshaped matrix along with the original number of dimensions.

        :param buf: Input buffer containing I/Q samples.
        :type buf: Array-like

        :return: matrix of shape (..., symbols, samples) and original number of dimensions.
        :rtype: Tuple[Array, int]
        """
        xp = self.xp
        import numpy as np

        if self.safe and isinstance(buf, np.ndarray):
            buf = xp.asarray(buf)      # move host→device if needed
        if buf.dtype != xp.complex64:
            buf = buf.astype(xp.complex64, copy=False)
        if buf.strides[-1] != buf.itemsize:
            buf = xp.ascontiguousarray(buf)

        # reshape so that last dim is sym_len
        base_shape = buf.shape
        if base_shape[-1] % self.sym_len:
            raise ValueError("buffer length not multiple of symbol length")
        new_tail = (-1, self.sym_len)
        mat = buf.reshape(*base_shape[:-1], *new_tail)
        return mat, buf.ndim          # keep original ndim for squeeze

    def _dechirp(self, mat, base_key):
        """
        Multiply with reference chirp (broadcast).
        
        This function applies the dechirping operation by multiplying the input matrix
        with the reference chirp corresponding to the specified base key. The chirp is
        retrieved or generated if it does not already exist in the reference dictionary.

        :param mat: Input matrix of shape (..., symbols, samples).
        :type mat: Array-like
        :param base_key: Key to identify the reference chirp, can be a string or LoRaMarkers.
        :type base_key: str or LoRaMarkers

        :return: Dechirped matrix of shape (..., symbols, samples).
        :rtype: Array
        
        """
        base = self._get_base(base_key)[None, None, :]
        return mat * base

    def _get_base(self, key):
        """
        Get or generate the base chirp for the given key.
        
        This function retrieves the reference chirp from the internal dictionary or generates it
        if it does not exist. The chirp is generated based on the specified slope and duration
        factor, which can be derived from the key if it is a `LoRaMarkers` instance or a string
        indicating a standard chirp type.

        :param key: Key to identify the reference chirp, can be a string or LoRaMarkers.
        :type key: str or LoRaMarkers

        :return: Reference chirp array.
        :rtype: Array
        """
        if key not in self._ref:
            if isinstance(key, LoRaMarkers):
                slope = key.slope_sign
                dur   = key.duration_factor
            elif key == "downchirp":
                slope, dur = -1, 1.0
            elif key == "upchirp":
                slope, dur = +1, 1.0
            else:
                raise ValueError("Unknown base chirp")

            self._ref[key] = generate_base_chirp(
                self.xp, self.phy_params, slope=slope, duration_factor=dur
            )
        return self._ref[key]

    def _maybe_squeeze(self, arr, orig_ndim):
        """Return 1-D if caller provided 1-D; maintain leading dims otherwise."""
        if orig_ndim == 1:
            return arr.reshape(-1)
        return arr

    # ------------------------- repr ------------------------------------
    def __repr__(self):
        return (f"<LoRaDemodulator backend={self.backend.name} "
                f"SF={self.phy_params.spreading_factor} BW={self.phy_params.bandwidth/1e3:.0f}k "
                f"SPC={self.phy_params.samples_per_chip}>")

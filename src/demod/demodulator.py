from __future__ import annotations
from typing import Any, Tuple
from dataclasses import dataclass
from src.core.backend   import choose_backend, AbstractBackend
from src.core.primitives import generate_base_chirp
from src.core.params    import LoRaPhyParams, LoRaFoldMode
from src.core.markers   import LoRaMarkers


K_PHIS = 16

def fold_fpa(xp, fft_, first, last, debug=False):
    """
    Full Phase Alignment (FPA) folding via brute-force phase search.
    """
    a = fft_[..., first]
    b = fft_[..., last]

    phis = xp.linspace(0.0, 2 * xp.pi, K_PHIS, endpoint=False, dtype=a.dtype)
    phasors = xp.exp(1j * phis)

    folded_all = a[..., None, :] + b[..., None, :] * phasors[None, :, None]

    mags = xp.abs(folded_all)
    peak_mags = xp.max(mags, axis=-1)
    best_idx = xp.argmax(peak_mags, axis=-1)

    phi_opt = phis[best_idx]

    if debug:
        import numpy as np
        if xp.__name__ == 'cupy':
            phi_opt_np = phi_opt.get()
            peak_mags_np = peak_mags.get()
            phis_np = phis.get()
        else: 
            phi_opt_np = phi_opt
            peak_mags_np = peak_mags
            phis_np = phis
        
        phi_opt_squeezed = np.squeeze(phi_opt_np)
        peak_mags_squeezed = np.squeeze(peak_mags_np)

        phi_opt_final = np.atleast_1d(phi_opt_squeezed)
        peak_mags_final = np.atleast_2d(peak_mags_squeezed)

        num_symbols = peak_mags_final.shape[0]

        for symbol_idx in range(num_symbols):
            winning_phase = float(np.real(phi_opt_final[symbol_idx]))
            print(f"\n--- Analysis for Symbol---")
            print(f"WINNING PHASE: {winning_phase:.4f} rad")
            print("Comparison of all candidate phases:")
            print("  Phase (rad)  | Peak Magnitude")
            print("---------------------------------")
            
            magnitudes_for_symbol = peak_mags_final[symbol_idx]

            for i in range(K_PHIS):
                candidate_phase = float(np.real(phis_np[i]))
                magnitude = float(magnitudes_for_symbol[i])
                marker = "<-- WINNER" if np.isclose(candidate_phase, winning_phase) else ""
                print(f"  {candidate_phase:<12.4f} | {magnitude:.4f} {marker}")

    folded = a + b * xp.exp(1j * phi_opt[..., None])
    return folded


@dataclass(frozen=True, slots=True)
class _FoldSpec:
    """
    Encapsulates oversampling-folding strategies used in FFT-based 
    LoRa demodulation to combine phase-aligned chips.
    """
    mode: LoRaFoldMode
    spc: int
    chips: int
    padding: int

    def build(self, xp, debug):
        """
        Returns a folding function according to the selected strategy.
        """
        # Effective bandwidth bins after padding
        bw_bins = self.chips * self.padding
        
        if self.spc == 1:
            return lambda fft_: fft_[..., :bw_bins]

        first = slice(0, bw_bins)
        # Grab the alias from the end of the spectrum (negative indexing handles the total length)
        last  = slice(-bw_bins, None) 

        if self.mode == LoRaFoldMode.OPA:
            return lambda fft_: fft_[..., first] + fft_[..., last]

        if self.mode == LoRaFoldMode.FPA:
            return lambda fft_: fold_fpa(xp, fft_, first, last, debug)

        if self.mode == LoRaFoldMode.CPA:
            return lambda fft_: xp.abs(fft_[..., first]) + xp.abs(fft_[..., last])

        raise ValueError(f"Unknown folding mode: {self.mode!r}")


class LoRaDemodulator:
    """
    Unified GPU/CPU LoRa demodulator.
    """

    # ------------------------- construction -----------------------------
    def __init__(self,
                 phy_params: LoRaPhyParams,
                 *,
                 backend: str | AbstractBackend = "auto",
                 fold_mode: LoRaFoldMode = LoRaFoldMode.OPA,
                 safe: bool = True,
                 debug_fpa: bool = False
                ):

        self.backend: AbstractBackend = (
            backend if isinstance(backend, AbstractBackend)
            else choose_backend(backend)
        )
        self.xp = self.backend.xp
        self.phy_params = phy_params
        self.safe = safe
        self.fold_mode = fold_mode
        self.debug_fpa = debug_fpa

        # --- constants --------------------------------------------------
        self.chips   = 1 << phy_params.spreading_factor
        self.sym_len = self.chips * phy_params.samples_per_chip

        # reference chirps on this backend
        self._ref: dict[str, Any] = {}
        self._get_base("downchirp")   # pre-cache

    def demodulate(self,
                    buf,
                    *,
                    base: str | LoRaMarkers = "downchirp",
                    padding: int = 1,
                    return_items: list[str] = ["symbols"]
                    ) -> tuple:
            """
            Demodulate a complex baseband waveform into LoRa symbol indices.
            """
            ALLOWED = {"symbols", "peaks", "folded", "deltas", "viz_bundle"}
            if not set(return_items).issubset(ALLOWED):
                raise ValueError(f"Invalid return_items: {return_items}")

            mat, orig_ndim = self._prepare(buf)               # shape (..., sym, samp)
            dech = self._dechirp(mat, base)                   # elementwise *
            
            fft_len = self.sym_len * padding
            fft  = self.backend.fft(dech, n=fft_len, axis=-1)
            
            # Build folding strategy dynamically based on padding
            fold_fn = _FoldSpec(
                mode=self.fold_mode,
                spc=self.phy_params.samples_per_chip,
                chips=self.chips,
                padding=padding
            ).build(self.xp, self.debug_fpa)
            
            folded = fold_fn(fft)                               # shape (..., sym, chips*padding)
            mag   = self.xp.abs(folded)                         # shape (..., sym, chips*padding)
            
            raw_idx = self.xp.argmax(mag, axis=-1)              # (..., sym) [INTEGER]

            actual_sym_val = self.xp.round(raw_idx / padding).astype(int)

            symbols_out = self._maybe_squeeze(actual_sym_val, orig_ndim)
            out = [symbols_out]

            if "peaks" in return_items:
                peaks = self.xp.take_along_axis(mag, raw_idx[..., None], axis=-1).squeeze(-1)
                peaks_out = self._maybe_squeeze(peaks, orig_ndim)
                out.append(peaks_out)
            
            if "folded" in return_items:
                folded_out = self._maybe_squeeze(mag, orig_ndim)
                out.append(folded_out)

            if "deltas" in return_items:
                idx_l = (raw_idx - 1) % mag.shape[-1]
                idx_r = (raw_idx + 1) % mag.shape[-1]
                
                mag_l = self.xp.take_along_axis(mag, idx_l[..., None], axis=-1).squeeze(-1)
                mag_c = self.xp.take_along_axis(mag, raw_idx[..., None], axis=-1).squeeze(-1) 
                mag_r = self.xp.take_along_axis(mag, idx_r[..., None], axis=-1).squeeze(-1)
                
                denom = (mag_l - 2 * mag_c + mag_r)
                denom_safe = self.xp.where(self.xp.abs(denom) < 1e-12, self.xp.nan, denom)
                
                # Delta in units of "padded bins"
                delta_padded_bins = 0.5 * (mag_l - mag_r) / denom_safe
                
                delta_syms = delta_padded_bins / padding

                deltas = self._maybe_squeeze(delta_syms, orig_ndim)
                out.append(deltas)

            if "viz_bundle" in return_items:
                peaks = self.xp.take_along_axis(mag, raw_idx[..., None], axis=-1).squeeze(-1)
                peaks_out = self._maybe_squeeze(peaks, orig_ndim)
                mag_out = mag.reshape(-1, mag.shape[-1])
                debug = {
                    "phy_params": self.phy_params,
                    "padding": padding,
                    "modulated_symbols": buf.get() if hasattr(buf, "get") else buf,
                    "demodulated_symbols": symbols_out.get() if hasattr(symbols_out, "get") else symbols_out,
                    "peak_magnitudes": peaks_out.get() if hasattr(peaks_out, "get") else peaks_out,
                    "folded_mag_fft": mag_out.get() if hasattr(mag_out, "get") else mag_out
                }
                out.append(debug)

            return tuple(out) if len(out) > 1 else out[0]

    # ------------------------- helpers ---------------------------------
    def _prepare(self, buf):
        xp = self.xp
        import numpy as np

        if self.safe and isinstance(buf, np.ndarray):
            buf = xp.asarray(buf)
        if buf.dtype != xp.complex64:
            buf = buf.astype(xp.complex64, copy=False)
        if buf.strides[-1] != buf.itemsize:
            buf = xp.ascontiguousarray(buf)

        base_shape = buf.shape
        if base_shape[-1] % self.sym_len:
            raise ValueError("buffer length not multiple of symbol length")
        new_tail = (-1, self.sym_len)
        mat = buf.reshape(*base_shape[:-1], *new_tail)
        return mat, buf.ndim

    def _dechirp(self, mat, base_key):
        base = self._get_base(base_key)[None, None, :]
        return mat * base

    def _get_base(self, key):
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
        if orig_ndim == 1:
            return arr.reshape(-1)
        return arr

    def __repr__(self):
        return (f"<LoRaDemodulator backend={self.backend.name} "
                f"SF={self.phy_params.spreading_factor} BW={self.phy_params.bandwidth/1e3:.0f}k "
                f"SPC={self.phy_params.samples_per_chip}>")

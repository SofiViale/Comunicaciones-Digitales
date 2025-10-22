"""LoRaModulator
================================================

Generate LoRa base-band I/Q samples from symbol sequences.

This class turns a sequence of LoRa **symbols** (plain integers) and/or
**markers** (`LoRaMarkers`) into a complex base-band waveform.
It reuses the stateless helpers in `core.primitives` and the backend
abstraction in `core.backend`, so it works unchanged on NumPy (CPU) or CuPy (GPU).
"""
from __future__ import annotations

from typing import Sequence, List, Union, Any

from src.core.params import LoRaPhyParams, LoRaFrameParams
from src.core.markers import LoRaMarkers
from src.core.backend import choose_backend, AbstractBackend
from src.core.primitives import (
    generate_base_chirp,
    instantaneous_phase,
    to_complex,
    instantaneous_frequency,
)

Symbol = Union[int, LoRaMarkers]


class LoRaModulator:
    """LoRa base-band modulator (I/Q waveform generator).

    :param phy_params: Immutable physical-layer parameters (SF, BW, oversampling).
    :type  phy: LoRaPhyParams
    :param frame_params: Frame-level knobs (preamble length, header, sync word).
    :type  frame: LoRaFrameParams
    :param backend: Backend engine ("auto", "numpy", "cupy" or custom).
    :type  backend: str | AbstractBackend
    :param enable_logging: Emit print messages during operation.
    :type  enable_logging: bool
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        phy_params: LoRaPhyParams,
        frame_params: LoRaFrameParams,
        *,
        backend: str | AbstractBackend | None = "auto",
        enable_logging: bool = False,
    ) -> None:
        self.phy_params = phy_params
        self.frame_params = frame_params
        self.backend: AbstractBackend = (
            backend if isinstance(backend, AbstractBackend) else choose_backend(backend)
        )
        self.xp = self.backend.xp
        self._log_enabled = enable_logging

        # --- constants -------------------------------------------------
        F = self.xp.float64
        self._coef = F(1.0) / self.xp.sqrt(
            F(phy_params.chips_per_symbol * phy_params.samples_per_chip)
        )

        # cached building blocks
        self._preamble_markers: List[LoRaMarkers] = [
            LoRaMarkers.FULL_UPCHIRP for _ in range(self.frame_params.preamble_symbol_count)
        ]
        self._sync_word: List[LoRaMarkers] = self.frame_params.encode_sync_word(self.phy_params.chips_per_symbol)
        self._sfd_markers: List[LoRaMarkers] = [
            LoRaMarkers.FULL_DOWNCHIRP,
            LoRaMarkers.FULL_DOWNCHIRP,
            LoRaMarkers.QUARTER_DOWNCHIRP,
        ]

    # ------------------------------------------------------------------ helpers
    def _log(self, msg: str) -> None:
        if self._log_enabled:
            print(f"[LoRaMod] {msg}")

    # .................................................................
    def _encode_length(self, length_sym: int) -> List[int]:
        """Map *payload length* to **two** SF-bit symbols ``[hi, lo]``."""
        sf = self.phy_params.spreading_factor
        if length_sym >= (1 << (2 * sf)):
            raise ValueError("Payload too long for current SF")
        mask = (1 << sf) - 1
        lo = length_sym & mask
        hi = (length_sym >> sf) & mask
        return [hi, lo]

    # .................................................................
    def _build_symbol_stream(
        self,
        payload: Sequence[Symbol],
        *,
        include_frame: bool,
        explicit_header: bool,
    ) -> List[Symbol]:
        """Assemble full transmission symbol sequence."""
        if not include_frame:
            return list(payload)

        header: List[int] = []
        if explicit_header and self.frame_params.explicit_header:
            header = self._encode_length(len(payload))



        return (
            self._preamble_markers
            + self._sync_word
            + self._sfd_markers
            + header 
            + list(payload)
        )

    # .................................................................
    def _mod_single_symbol(self, sym: Symbol):
        """Return complex64 array for *one* symbol/marker."""
        if isinstance(sym, LoRaMarkers):
            return generate_base_chirp(
                self.xp,
                self.phy_params,
                slope=sym.slope_sign,
                duration_factor=sym.duration_factor,
            )

        # Plain integer → standard up-chirp of full duration
        phase = instantaneous_phase(self.xp, [int(sym)], self.phy_params)
        return to_complex(self.xp, phase, self._coef)

    # ------------------------------------------------------------------ public
    def modulate(
        self,
        payload: Sequence[Symbol],
        *,
        debug_bundle: bool = False,
        include_frame: bool = True,
        explicit_header: bool | None = None,
    ):
        """
        Modulate a sequence of LoRa symbols into a complex baseband waveform.

        This method converts a list of symbols (integers or markers) into a 
        discrete-time complex waveform representing LoRa-modulated I/Q samples. 
        Optionally includes a LoRa-compatible preamble, sync word, SFD, and explicit 
        header.

        When ``debug_bundle`` is enabled, it also returns auxiliary diagnostics 
        including instantaneous frequency and frame section boundaries.

        :param payload: Sequence of symbols to modulate. Symbols may be integers 
                        (for data) or :class:`LoRaMarkers` (for control).
        :type payload: Sequence[Symbol]

        :param debug_bundle: If ``True``, return a dictionary of debug information 
                             alongside the waveform.
        :type debug_bundle: bool, optional

        :param include_frame: If ``True``, prepend the preamble, sync word, SFD, and 
                              optionally the explicit header.
        :type include_frame: bool, optional

        :param explicit_header: Override the use of an explicit header (2-symbol 
                                length field). If ``None``, uses the frame default.
        :type explicit_header: bool or None, optional

        :returns: 
            - waveform (complex64 array): Modulated signal.
            - debug_info (dict): Only if ``debug_bundle`` is ``True``. Contains keys:
                ``"signal"``, ``"instantaneous_frequency"``, ``"time_axis"``, and 
                frame section boundaries like ``"preamble_end"``, etc.
        :rtype: tuple[Array, dict] or Array
        """
        if explicit_header is None:
            explicit_header = self.frame_params.explicit_header

        # 1 ─ Build full symbol sequence ---------------------------------
        symbols: List[Symbol] = self._build_symbol_stream(
            payload,
            include_frame=include_frame,
            explicit_header=explicit_header,
        )

        self._log(f"Generating {len(symbols)} symbols (backend={self.backend.name})")

        # 2 ─ Concatenate waveforms --------------------------------------
        parts = [self._mod_single_symbol(s) for s in symbols]
        sig = self.xp.concatenate(parts)

        # 3a ─ Fast path --------------------------------------------------
        if not debug_bundle:
            return sig.astype(self.xp.complex64, copy=False)
        
        # 3b ─ Full diagnostics bundle -----------------------------------
        t = self.xp.arange(sig.size, dtype=self.xp.float32) * self.phy_params.sample_duration
        freq = instantaneous_frequency(self.xp, symbols, self.phy_params)

        bundle = {
            "mode": "frame" if include_frame else "payload_only",
            "payload_symbols": list(payload),
            "spreading_factor": self.phy_params.spreading_factor,
            "bandwidth": self.phy_params.bandwidth,
            "samples_per_chip": self.phy_params.samples_per_chip,
            "signal": sig,
            "time_axis": t,
            "instantaneous_frequency": freq,
        }

        if include_frame:
            sps = self.phy_params.samples_per_symbol
            pre_end = self.frame_params.preamble_symbol_count * sps - 1
            sfd_end = int((self.frame_params.preamble_symbol_count + 4.25) * sps) - 1
            hdr_end = (
                int((self.frame_params.preamble_symbol_count + 6.25) * sps) - 1
                if explicit_header else None
            )
            bundle.update({
                "has_explicit_header": bool(explicit_header),
                "preamble_symbol_count": self.frame_params.preamble_symbol_count,
                "indexes": {
                    "preamble_end": pre_end,
                    "sfd_end":      sfd_end,
                    "header_end":   hdr_end,
                },
            })
        return sig.astype(self.xp.complex64, copy=False), bundle 

    def generate_sync_base(self):
        """
        Generate the synchronization base for the given PHY and frame parameters.

        This method constructs the base sequence used for synchronization, which
        includes the preamble markers, sync word, and SFD markers.
        
        :returns: The generated synchronization base as a complex64 array.
        :rtype: np.ndarray
        """
        # Assemble the full symbol sequence for the sync base
        symbols = self._build_symbol_stream(
            payload=[],
            include_frame=True,
            explicit_header=False,  # No header in sync base
        )
        
        # Modulate the symbols into a complex waveform
        waveform = self.xp.concatenate([self._mod_single_symbol(s) for s in symbols])
        
        return waveform.astype(self.xp.complex64, copy=False)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        return (
            f"LoRaModulator(SF={self.phy_params.spreading_factor}, "
            f"BW={self.phy_params.bandwidth/1e3:.0f}kHz, SPC={self.phy_params.samples_per_chip}, "
            f"PSC={self.frame_params.preamble_symbol_count}, backend={self.backend.name})"
        )

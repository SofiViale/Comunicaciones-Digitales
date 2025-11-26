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
    locate_discontinuity_indices
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

        F = self.xp.float64
        self._coef = F(1.0) / self.xp.sqrt(
            F(phy_params.chips_per_symbol * phy_params.samples_per_chip)
        )

        self._data_symbol_base = self._precompute_data_base()

        self._preamble_markers: List[LoRaMarkers] = [
            LoRaMarkers.FULL_UPCHIRP for _ in range(self.frame_params.preamble_symbol_count)
        ]
        self._sync_word: List[LoRaMarkers] = self.frame_params.encode_sync_word(self.phy_params.chips_per_symbol)
        self._sfd_markers: List[LoRaMarkers] = [
            LoRaMarkers.FULL_DOWNCHIRP,
            LoRaMarkers.FULL_DOWNCHIRP,
            LoRaMarkers.QUARTER_DOWNCHIRP,
        ]

    def _log(self, msg: str) -> None:
        if self._log_enabled:
            print(f"[LoRaMod] {msg}")

    def _precompute_data_base(self) -> Any:
        """Generates two concatenated base up-chirps (symbol 0) for fast slicing."""
        phase_0 = instantaneous_phase(self.xp, [0, 0], self.phy_params)
        return to_complex(self.xp, phase_0, self._coef)

    def _encode_length(self, length_sym: int) -> List[int]:
        sf = self.phy_params.spreading_factor
        if length_sym >= (1 << (2 * sf)):
            raise ValueError("Payload too long for current SF")
        mask = (1 << sf) - 1
        lo = length_sym & mask
        hi = (length_sym >> sf) & mask
        return [hi, lo]

    def _build_symbol_stream(
        self,
        payload: Sequence[Symbol],
        *,
        include_frame: bool,
        explicit_header: bool,
    ) -> List[Symbol]:
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

    def _mod_marker_symbol(self, sym: LoRaMarkers) -> Any:
        """Return complex64 array for one LoRaMarkers instance."""
        return generate_base_chirp(
            self.xp,
            self.phy_params,
            slope=sym.slope_sign,
            duration_factor=sym.duration_factor,
        )

    def modulate(
        self,
        payload: Sequence[Symbol],
        *,
        legacy: bool = False,
        debug_bundle: bool = False,
        include_frame: bool = True,
        explicit_header: bool | None = None
    ):
        """
        Modulate a sequence of LoRa symbols into a complex baseband waveform.

        This method acts as a router to select between a legacy, formula-based
        modulator and a faster, buffer-based optimized version.

        :param payload: Sequence of symbols to modulate.
        :type payload: Sequence[Symbol]
        :param legacy: If `True`, use the original formula-based modulation method.
                       Defaults to `False`, using the optimized method.
        :type legacy: bool, optional
        :param debug_bundle: If `True`, return a dictionary of debug information.
        :type debug_bundle: bool, optional
        :param include_frame: If `True`, prepend the preamble, sync word, and SFD.
        :type include_frame: bool, optional
        :param explicit_header: Override the use of an explicit header.
        :type explicit_header: bool or None, optional
        :returns: Waveform array, or a tuple of (waveform, debug_info).
        """
        if legacy:
            return self._legacy_modulate(
                payload,
                debug_bundle=debug_bundle,
                include_frame=include_frame,
                explicit_header=explicit_header
            )
        else:
            return self._optimized_modulate(
                payload,
                debug_bundle=debug_bundle,
                include_frame=include_frame,
                explicit_header=explicit_header
            )

    def _legacy_modulate(
        self,
        payload: Sequence[Symbol],
        *,
        debug_bundle: bool = False,
        include_frame: bool = True,
        explicit_header: bool | None = None
    ):
        """Original modulation method: generates and concatenates each symbol individually."""
        if explicit_header is None:
            explicit_header = self.frame_params.explicit_header

        symbols: List[Symbol] = self._build_symbol_stream(
            payload,
            include_frame=include_frame,
            explicit_header=explicit_header,
        )

        self._log(f"Generating {len(symbols)} symbols (legacy backend={self.backend.name})")

        def _mod_single_symbol(sym: Symbol):
            if isinstance(sym, LoRaMarkers):
                return self._mod_marker_symbol(sym)
            phase = instantaneous_phase(self.xp, [int(sym)], self.phy_params)
            return to_complex(self.xp, phase, self._coef)

        parts = [_mod_single_symbol(s) for s in symbols]
        sig = self.xp.concatenate(parts)

        if not debug_bundle:
            return sig.astype(self.xp.complex64, copy=False)

        # Full diagnostics bundle generation
        return self._create_debug_bundle(sig, symbols, payload, include_frame, explicit_header)

    def _optimized_modulate(
        self,
        payload: Sequence[Symbol],
        *,
        debug_bundle: bool = False,
        include_frame: bool = True,
        explicit_header: bool | None = None
    ):
        """Optimized modulation method: pre-allocates and fills a buffer using views."""
        if explicit_header is None:
            explicit_header = self.frame_params.explicit_header

        symbols: List[Symbol] = self._build_symbol_stream(
            payload,
            include_frame=include_frame,
            explicit_header=explicit_header,
        )

        self._log(f"Generating {len(symbols)} symbols (optimized backend={self.backend.name})")

        sps = self.phy_params.samples_per_symbol
        total_samples = sum(
            int(s.duration_factor * sps) if isinstance(s, LoRaMarkers) else sps
            for s in symbols
        )
        sig = self.xp.empty(total_samples, dtype=self.xp.complex64)

        current_pos = 0
        spc = self.phy_params.samples_per_chip

        for sym in symbols:
            if isinstance(sym, LoRaMarkers):
                waveform = self._mod_marker_symbol(sym)
                end_pos = current_pos + len(waveform)
                sig[current_pos:end_pos] = waveform
            else:
                shift_samples = int(sym) * spc
                start_index = shift_samples
                end_index = start_index + sps
                end_pos = current_pos + sps
                sig[current_pos:end_pos] = self._data_symbol_base[start_index:end_index]

            current_pos = end_pos

        if not debug_bundle:
            return sig.astype(self.xp.complex64, copy=False)

        # Full diagnostics bundle generation
        return self._create_debug_bundle(sig, symbols, payload, include_frame, explicit_header)

    def _create_debug_bundle(self, sig, symbols, payload, include_frame, explicit_header):
        """Helper to generate the diagnostic dictionary."""
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

    def locate_discontinuities(
            self,
            payload: Sequence[Symbol],
            *,
            include_frame: bool = True,
            explicit_header: bool | None = None,
        ):
        if explicit_header is None:
            explicit_header = self.frame_params.explicit_header

        symbols: List[Symbol] = self._build_symbol_stream(
            payload,
            include_frame=include_frame,
            explicit_header=explicit_header,
        )
        return locate_discontinuity_indices(symbols, self.phy_params)

    def generate_sync_base(self):
        """
        Generates the synchronization base (preamble, sync word, SFD).
        """
        return self.modulate(
            payload=[],
            legacy=False, # Always use the fast method for this fixed sequence
            include_frame=True,
            explicit_header=False,
        )

    def __repr__(self) -> str:
        return (
            f"LoRaModulator(SF={self.phy_params.spreading_factor}, "
            f"BW={self.phy_params.bandwidth/1e3:.0f}kHz, SPC={self.phy_params.samples_per_chip}, "
            f"PSC={self.frame_params.preamble_symbol_count}, backend={self.backend.name})"
        )

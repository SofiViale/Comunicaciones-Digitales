import numpy as np
from typing import Union, List
from src.demod.demodulator import LoRaDemodulator
from src.mod.modulator import LoRaModulator
from src.core.params import LoRaPhyParams, LoRaFrameParams
from src.core.backend import AbstractBackend, choose_backend


class CorrelationBasedSynchronizer:
    """
    Synchronizer based on cross-correlation with the preamble base.

    This synchronizer detects the start of LoRa frames using standard
    cross-correlation with a known synchronization pattern.
    """

    def __init__(self,
                 phy_params: LoRaPhyParams,
                 frame_params: LoRaFrameParams,
                 *,
                 backend: str | AbstractBackend = "auto",
                 correlation_threshold: float = 0.9):
        self.phy_params = phy_params
        self.frame_params = frame_params
        self.backend: AbstractBackend = choose_backend(backend)
        self.threshold = correlation_threshold

        self.samples_per_symbol = self.phy_params.samples_per_symbol

        self.sync_base = LoRaModulator(self.phy_params, self.frame_params).generate_sync_base()
        self.demod = LoRaDemodulator(
            phy_params=self.phy_params,
            backend=self.backend
        )

    def _cross_correlate(self, signal: Union[np.ndarray]) -> Union[np.ndarray]:
        """
        Cross-correlate the signal with the sync base using backend module.

        Uses the identity: correlate(x, y) = convolve(x, conj(reverse(y)))

        :param signal: Input signal buffer (1D array).
        :type signal: np.ndarray
        
        :return: Cross-correlation result.
        :rtype: np.ndarray
        """
        xp = self.backend.xp
        sync = xp.asarray(self.sync_base)
        sig = xp.asarray(signal)

        y_rev_conj = xp.conj(xp.flip(sync))
        corr = xp.convolve(sig, y_rev_conj, mode='full')
        return xp.abs(corr)

    def _find_candidate_indices(self, signal: Union[np.ndarray]) -> List[int]:
        """
        Get candidate synchronization start indices based on thresholding.

        :param signal: Input signal buffer (1D array).
        :type signal: np.ndarray

        :return: List of indices where the correlation exceeds the threshold.
        :rtype: List[int]
        """
        xp = self.backend.xp
        corr = self._cross_correlate(signal)
        max_corr = float(corr.max())
        threshold = self.threshold * max_corr
        offset = len(self.sync_base) - 1

        indices = xp.where(corr > threshold)[0]
        starts = [int(idx - offset) for idx in indices if (idx - offset) >= 0]

        return sorted(set(starts))

    def run(self, iq_samples: Union[np.ndarray], *, debug_bundle:bool = False) -> Union[np.ndarray, None]:
        """
        Detect the best synchronization point in the buffer and extract its payload.

        :param iq_samples: Input IQ samples buffer (1D array).
        :type iq_samples: np.ndarray

        :return: Extracted payload waveform, or None if detection fails.
        :rtype: np.ndarray or None
        """
        xp = self.backend.xp
        sps = self.samples_per_symbol
        total_len = iq_samples.shape[0]

        # Perform cross-correlation and find best peak
        corr = self._cross_correlate(iq_samples)
        best_idx = int(xp.argmax(corr))
        offset = len(self.sync_base) - 1
        start = best_idx - offset

        if start < 0 or (start + len(self.sync_base) > total_len):
            return None

        header_start = start + len(self.sync_base)
        if header_start + 2 * sps > total_len:
            return None

        header = iq_samples[header_start : header_start + 2 * sps]
        header_syms = self.demod.demodulate(header, base="downchirp")

        try:
            hi, lo = int(header_syms[0]), int(header_syms[1])
            payload_len = (hi << self.phy_params.spreading_factor) | lo
        except Exception:
            return None

        payload_start = header_start + 2 * sps
        payload_end = payload_start + payload_len * sps
        if payload_end > total_len:
            return None

        if not debug_bundle:
            return iq_samples[payload_start:payload_end]
        
        psc = self.frame_params.preamble_symbol_count

        debug_bundle = {
            "phy_params": self.phy_params,
            "iq_samples": iq_samples,
            "preamble_start_offset": start,
            "sync_word_start_offset": start + psc * sps,
            "sfd_start_offset": start + (psc + 2) * sps,
            "header_start_offset": header_start,
            "payload_start_offset": payload_start,
            "payload_end_offset": payload_end,
            "header_symbols": header_syms,
        }
        return iq_samples[payload_start:payload_end], debug_bundle


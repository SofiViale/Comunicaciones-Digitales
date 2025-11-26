import numpy as np
from typing import Union, List, Tuple
from src.demod.demodulator import LoRaDemodulator
from src.mod.modulator import LoRaModulator
from src.core.params import LoRaPhyParams, LoRaFrameParams, CorrelationSyncParams, LoRaFoldMode
from src.core.backend import AbstractBackend, choose_backend
import src.sync.exceptions as sync_exceptions

class CorrelationBasedSynchronizer:
    """
    Synchronizer based on cross-correlation with the preamble base.

    This synchronizer detects the start of LoRa frames using standard
    cross-correlation with a known synchronization pattern.
    """

    def __init__(self,
                 phy_params: LoRaPhyParams,
                 frame_params: LoRaFrameParams,
                 sync_params: CorrelationSyncParams,
                 fold_mode: LoRaFoldMode,
                 *,
                 backend: str | AbstractBackend = "auto"):
        self.phy_params = phy_params
        self.frame_params = frame_params
        self.backend: AbstractBackend = choose_backend(backend)
        self.threshold = sync_params.correlation_threshold
        self.max_sync_candidates = 50

        self.samples_per_symbol = self.phy_params.samples_per_symbol

        self.sync_base = LoRaModulator(self.phy_params, self.frame_params).generate_sync_base()
        self.demod = LoRaDemodulator(
            phy_params=self.phy_params,
            backend=self.backend,
            fold_mode=fold_mode,
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

        #y_rev_conj = xp.conj(xp.flip(sync))
        #corr = xp.convolve(sig, y_rev_conj, mode='full')
        corr = xp.correlate(sig, sync, mode='full')
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

    def run(self, 
            iq_samples: Union[np.ndarray], 
            *, 
            debug_bundle:bool = False
           ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Detect the best synchronization point in the buffer and extract its payload.
        Tries up to 'max_sync_candidates' peaks.

        :param iq_samples: Input IQ samples buffer (1D array).
        :type iq_samples: np.ndarray
        :param debug_bundle: If True, returns a tuple (payload, debug_dict).
        
        :raises sync_exceptions.SynchronizationError: If sync fails (no peak, short buffer, bad header).
        :raises sync_exceptions.CandidatesExhaustedError: If all candidates fail.
        :return: Extracted payload waveform (or tuple si debug_bundle=True).
        :rtype: np.ndarray or Tuple[np.ndarray, dict]
        """
        xp = self.backend.xp
        sps = self.samples_per_symbol
        total_len = iq_samples.shape[0]
        offset = len(self.sync_base) - 1

        # 1. Realizar la correlación UNA SOLA VEZ
        corr = self._cross_correlate(iq_samples)
        
        # 2. Encontrar todos los picos por encima del umbral
        max_corr = float(corr.max())
        if max_corr == 0:
             raise sync_exceptions.NoCandidatesFoundError("Correlation resulted in all zeros.")
             
        threshold_val = self.threshold * max_corr
        candidate_indices_corr = xp.where(corr > threshold_val)[0]
        
        if candidate_indices_corr.size == 0:
            raise sync_exceptions.NoCandidatesFoundError(
                f"No correlation peaks found above threshold ({threshold_val:.2f}). Max peak was {max_corr:.2f}."
            )

        # 3. Ordenar los picos candidatos por su fuerza (de mayor a menor)
        candidate_strengths = corr[candidate_indices_corr]
        sorted_by_strength_idx = xp.argsort(candidate_strengths)[::-1]
        
        best_indices_ordered = candidate_indices_corr[sorted_by_strength_idx]
        
        # 4. Limitar al número máximo de candidatos
        num_to_try = min(len(best_indices_ordered), self.max_sync_candidates)
        final_candidate_list = best_indices_ordered[:num_to_try]
        
        # 5. Iterar sobre los candidatos y probar la sincronización
        for i, best_idx in enumerate(final_candidate_list):
            try:
                start = int(best_idx) - offset

                if start < 0 or (start + len(self.sync_base) > total_len):
                    raise sync_exceptions.SynchronizationError(
                        f"Candidate {i} (idx={best_idx}) results in out-of-bounds start {start}."
                    )

                header_start = start + len(self.sync_base)
                if header_start + 2 * sps > total_len:
                    raise sync_exceptions.IncompleteHeaderError(
                        f"Candidate {i} (start={start}) leaves no room for header."
                    )

                header = iq_samples[header_start : header_start + 2 * sps]
                header_syms = self.demod.demodulate(header, base="downchirp")

                try:
                    hi, lo = int(header_syms[0]), int(header_syms[1])
                    payload_len = (hi << self.phy_params.spreading_factor) | lo
                except (IndexError, TypeError, ValueError) as e:
                    raise sync_exceptions.SynchronizationError(
                        f"Candidate {i} (start={start}) failed to decode header symbols {header_syms}. Error: {e}"
                    )

                payload_start = header_start + 2 * sps
                payload_end = payload_start + payload_len * sps
                available_samples = total_len - payload_start
                
                if payload_end > total_len:
                    raise sync_exceptions.IncompletePayloadError(
                        f"Candidate {i} (start={start}): Buffer too short. Header indicates {payload_len} symbols "
                        f"({payload_len * sps} samples), but only "
                        f"{available_samples} samples are available."
                    )
                
                # Si llegamos aquí, este candidato es bueno.
                payload = iq_samples[payload_start:payload_end]

                if not debug_bundle:
                    return payload
                
                psc = self.frame_params.preamble_symbol_count

                debug_bundle_dict = {
                    "phy_params": self.phy_params,
                    "iq_samples": iq_samples,
                    "preamble_start_offset": start,
                    "sync_word_start_offset": start + psc * sps,
                    "sfd_start_offset": start + (psc + 2) * sps,
                    "header_start_offset": header_start,
                    "payload_start_offset": payload_start,
                    "payload_end_offset": payload_end,
                    "header_symbols": header_syms.get() if hasattr(header_syms, 'get') else header_syms,
                }
                return payload, debug_bundle_dict

            except (sync_exceptions.SynchronizationError, sync_exceptions.IncompleteFrameError) as e:
                # Este candidato falló. El bucle continuará con el siguiente.
                continue 
        
        # 6. Si el bucle termina, todos los candidatos fallaron.
        raise sync_exceptions.CandidatesExhaustedError(
            f"All {num_to_try} correlation candidates were exhausted without successful synchronization."
        )

import numpy as np
from typing import Union, Tuple, List, Optional
from dataclasses import dataclass, field

# Constant Threshold expressions
CFO_THRESHOLD = 0.1 
sfo_threshold= lambda sf, spc: (1<<sf) / (1000 * spc)

# ------------------------------
# Error types (unchanged)
# ------------------------------
class SynchronizationError(Exception):
    """Base class for synchronization-related errors."""
    pass

class NoCandidatesFoundError(SynchronizationError):
    """No synchronization candidates were found."""
    pass

class SFDError(SynchronizationError):
    """Failed to locate the downchirp pair in SFD."""
    pass

class IncompleteFrameError(SynchronizationError):
    """IQ buffer too short to extract the body."""
    pass

class IncompleteHeaderError(IncompleteFrameError):
    """IQ buffer too short to extract the header."""
    pass

class IncompletePayloadError(IncompleteFrameError):
    """IQ buffer too short to extract as many payload symbols as the header indicates."""
    pass

class CandidatesExhaustedError(SynchronizationError):
    """All candidates were exhausted without successful synchronization."""
    pass

# Backend now provides as_strided; CuPy import is optional for type hints only.
try:
    import cupy as cp  # optional, used only for type annotations
except ImportError:
    cp = None



# ------------------------------
# Internal deps
# ------------------------------
from src.demod.demodulator import LoRaDemodulator
from src.core.params import LoRaPhyParams, LoRaFrameParams
from src.core.backend import AbstractBackend, choose_backend


# ------------------------------
# Data structures
# ------------------------------
@dataclass
class RunCandidate:
    row_idx: int
    col_idx: int  # symbol index in symbols_2d
    run_len: int
    symbol: int
    run_score: float
    total_score: float

    def __repr__(self):
        return (f"RunCandidate(row_idx={self.row_idx}, col_idx={self.col_idx}, "
                f"run_len={self.run_len}, symbol={self.symbol}, "
                f"run_score={self.run_score}, total_score={self.total_score})")

    def __str__(self):
        return (f"RunCandidate(row={self.row_idx}, col={self.col_idx}, "
                f"len={self.run_len}, sym={self.symbol}, "
                f"run_score={self.run_score:.2f}, total_score={self.total_score:.2f})")

@dataclass
class CandidateTrace:
    """Per-candidate debugging info (returned only when debug=True)."""
    index: int
    candidate: RunCandidate
    sync_offset: Optional[int] = None
    sfd_end_offset: Optional[int] = None
    header_syms: Optional[Tuple[int, int]] = None
    decoded_payload_len: Optional[int] = None
    needed_samples: Optional[int] = None
    available_samples: Optional[int] = None
    status: str = "error"  # "error" | "success"
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    # ---------- pretty printing helpers ----------
    @staticmethod
    def _fmt(val, default: str = "-"):
        return default if val is None else str(val)

    @staticmethod
    def _fmt_pair(pair: Optional[Tuple[int, int]], default: str = "-"):
        if pair is None:
            return default
        hi, lo = pair
        return f"({hi},{lo})"

    @staticmethod
    def _shorten(text: Optional[str], max_len: int = 120):
        if not text:
            return "-"
        s = str(text).replace("\n", " ")
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    def __str__(self) -> str:
        """Human-friendly, multi-line summary of the candidate trace."""
        icon = "✔" if self.status == "success" else "✖"
        c = self.candidate  # RunCandidate
        # One-line headline with candidate ranking + scoring
        headline = (
            f"[{icon}] Candidate {self.index}  "
            f"(row={c.row_idx}, col={c.col_idx}, sym={c.symbol}, "
            f"run_len={c.run_len}, run={c.run_score:.3f}, total={c.total_score:.3f})"
        )
        # Secondary lines
        line1 = (
            f"    sync_offset={self._fmt(self.sync_offset)}  "
            f"sfd_end={self._fmt(self.sfd_end_offset)}  "
            f"header_syms={self._fmt_pair(self.header_syms)}  "
            f"payload_len={self._fmt(self.decoded_payload_len)}"
        )
        line2 = (
            f"    needed={self._fmt(self.needed_samples)}  "
            f"available={self._fmt(self.available_samples)}"
        )
        err = "-" if not self.error_type else f"{self.error_type}: {self._shorten(self.error_msg)}"
        notes = " | ".join(self.notes) if self.notes else "-"
        line3 = f"    error={err}"
        line4 = f"    notes: {self._shorten(notes)}"
        return "\n".join([headline, line1, line2, line3, line4])
# ---------------------------------------------------------------------------
# DechirpBasedSynchronizer class
# ---------------------------------------------------------------------------

class DechirpBasedSynchronizer:
    def __init__(self,
                 phy_params: LoRaPhyParams,
                 frame_params: LoRaFrameParams,
                 *,
                 backend: str | AbstractBackend = "auto",
                 fold_mode = "0FPA",
                 max_sync_candidates: int = 2,
                 logging: bool = False,
                 compensate_cfo_sfo: bool = True,
                 debug: bool = False):
        
        self.phy_params = phy_params
        self.frame_params = frame_params
        self.backend: AbstractBackend = choose_backend(backend)
        self.max_sync_candidates = max_sync_candidates
        self.logging = logging
        self.compensate_cfo_sfo = compensate_cfo_sfo
        self.debug = debug

        self.demod = LoRaDemodulator(
            phy_params= self.phy_params,
            backend=self.backend,
            fold_mode=fold_mode,
        )

    def _log(self, caller:str, message: str):
        """Log messages with caller context."""
        if self.logging:
            print(f"[DechirpBasedSync - {caller}] {message}")

    def _generate_subchip_view(self, iq_samples: Union[np.ndarray, "cp.ndarray"]) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Create a 2D view of the IQ samples with sub-chip sample shifts.
        Each row is a shifted version of the IQ buffer, 0..(samples_per_chip - 1).
        """
        xp = self.backend.xp
        samples_per_chip = self.phy_params.samples_per_chip
        spreading_factor = self.phy_params.spreading_factor

        symbol_len = samples_per_chip * (1 << spreading_factor)
        phase_rows = samples_per_chip

        buf = xp.asarray(iq_samples)
        available = buf.size - phase_rows + 1
        window_len = (available // symbol_len) * symbol_len
        if window_len == 0:
            raise ValueError("IQ buffer too short for one complete symbol")

        # TODO: push `as_strided` into AbstractBackend for backend-neutral view creation.
        return self.backend.as_strided(
            buf,
            shape=(phase_rows, window_len),
            strides=(buf.itemsize, buf.itemsize),
        )

    def _find_preamble_candidates(self, iq_samples: Union[np.ndarray, "cp.ndarray"]) -> List[RunCandidate]:
        """
        Find top-N preamble candidates. Scoring is encapsulated here (no extra helper).
        Logic matches previous best_repeated_runs, just embedded.
        """
        expanded = self._generate_subchip_view(iq_samples)
        up_syms, up_peaks = self.demod.demodulate(expanded, base="downchirp", return_items=["symbols", "peaks"])
        up_syms = up_syms.get() if hasattr(up_syms, "get") else up_syms
        up_peaks = up_peaks.get() if hasattr(up_peaks, "get") else up_peaks

        preamble_symbol_count = self.frame_params.preamble_symbol_count
        sf = self.phy_params.spreading_factor
        encoded_sync_word = tuple(self.frame_params.encode_sync_word(sf))

        ideal_upchirps = preamble_symbol_count + (2 if encoded_sync_word == (0, 0) else 0)
        MAX_RUN_LEN = ideal_upchirps * 2
        min_run_len = max(preamble_symbol_count-1, preamble_symbol_count // 2)

        candidates: List[RunCandidate] = []

        for row_idx, (row_syms, row_mags) in enumerate(zip(up_syms, up_peaks)):
            start = np.array([row_syms[0] - 1], dtype=row_syms.dtype)
            end = np.array([row_syms[-1] + 1], dtype=row_syms.dtype)
            changes = np.flatnonzero(np.diff(np.concatenate((start, row_syms, end))))
            starts, ends = changes[:-1], changes[1:] - 1
            lengths = ends - starts + 1

            for s, e, run_len in zip(starts.tolist(), ends.tolist(), lengths.tolist()):
                if run_len < min_run_len or run_len > MAX_RUN_LEN:
                    continue

                symbol = (-int(row_syms[s])) & ((1 << sf) - 1)
                if e >= s:
                    score_slice = row_mags[s:e + 1]
                else:
                    continue

                if score_slice.size == 0:
                    continue

                run_score = float(score_slice.mean())

                # we finalize total_score after normalization
                candidates.append(RunCandidate(
                    row_idx=row_idx,
                    col_idx=s,
                    run_len=run_len,
                    symbol=symbol,
                    run_score=run_score,
                    total_score=0.0,
                ))

        if not candidates:
            raise NoCandidatesFoundError("No preamble candidates found.")

        # Normalize run scores 
        run_scores = np.array([c.run_score for c in candidates])
        min_score = run_scores.min()
        ptp_score = np.ptp(run_scores) + 1e-8
        norm_scores = (run_scores - min_score) / ptp_score

        for cand, norm_run_score in zip(candidates, norm_scores):
            run_len_score = 1.0 - abs(cand.run_len - ideal_upchirps) / ideal_upchirps
            run_len_score = max(run_len_score, 0.0)
            cand.total_score = 0.7 * norm_run_score + 0.3 * run_len_score

        # Return top-N
        return sorted(candidates, key=lambda c: -c.total_score)[: self.max_sync_candidates]

    def _sync_offset_from_candidate(self, cand: RunCandidate) -> int:
        """ Calculates where to start the IQ buffer to align to the frame start based on the candidate."""
        sub_chip_offset = cand.row_idx
        chip_offset = cand.symbol * self.phy_params.samples_per_chip
        symbol_offset = cand.col_idx * self.phy_params.samples_per_symbol
        return sub_chip_offset + chip_offset + symbol_offset

    def _apply_cfo_sfo_correction(self, iq, cfo_residue: float, sfo_residue: float):
        """Apply CFO/SFO phase compensation."""
        chips = 1 << self.phy_params.spreading_factor
        sps = chips * self.phy_params.samples_per_chip
        n = np.arange(iq.shape[0], dtype=np.float64)
        cfo_phase = 2.0 * np.pi * cfo_residue * n / chips
        sfo_phase = 2.0 * np.pi * sfo_residue * (n / sps) ** 2 / 2.0
        correction = np.exp(-1j * (cfo_phase + sfo_phase)).astype(iq.dtype, copy=False)
        return iq * correction

    def _estimate_and_maybe_apply_cfo_sfo(self, synced: Union[np.ndarray, "cp.ndarray"], trace: Optional[CandidateTrace] = None):
        """Estimate CFO/SFO from preamble and optionally apply corrections."""
        
        preamble_samples = self.frame_params.preamble_symbol_count * self.phy_params.samples_per_symbol
        preamble_segment = synced[:preamble_samples]
        if len(preamble_segment) < self.phy_params.samples_per_symbol:
            raise SynchronizationError("Preamble segment is too short for CFO/SFO estimation.")

        syms, deltas = self.demod.demodulate(preamble_segment, base="downchirp", return_items=["symbols", "deltas"])
        x = np.arange(len(syms.get() if hasattr(syms, "get") else syms))
        y = deltas.get() if hasattr(deltas, "get") else deltas
        sfo_residue, cfo_residue = np.polyfit(x, y, deg=1)  # slope, intercept
        cfo, sfo = cfo_residue, sfo_residue

        self._log("run", f"CFO/SFO residues: {cfo}, {sfo}")
        if trace is not None:
            trace.notes.append(f"cfo={cfo:.6f}, sfo={sfo:.6f}")

        should_apply_compensation = np.abs(cfo) > CFO_THRESHOLD or np.abs(sfo) > sfo_threshold(self.phy_params.spreading_factor, self.phy_params.samples_per_chip)

        if self.compensate_cfo_sfo and should_apply_compensation:
            self._log("run", f"Threshold met. Applying CFO/SFO correction with cfo={cfo}, sfo={sfo}")
            if trace is not None:
                trace.notes.append("CFO/SFO correction applied")
            synced = self._apply_cfo_sfo_correction(synced, cfo, sfo)
        return synced

    def _check_sfd(self, up_syms:np.ndarray, up_peaks:np.ndarray, down_syms:np.ndarray, down_peaks:np.ndarray) -> int:
        """
        Vectorised LoRa SFD locator.
        Returns the sample offset to the end of the SFD, or -1 if no (0, 0) pair meets mask criteria.
        """
        up_syms, up_peaks, down_syms, down_peaks = [
            a.get() if hasattr(a, "get") else np.asarray(a)
            for a in (up_syms, up_peaks, down_syms, down_peaks)
        ]
        n = up_syms.size
        if n < 3:
            return -1

        # mask: every ≥2‑long zero‑run in up_syms plus the next two positions
        z = up_syms == 0
        d = np.diff(np.pad(z.astype(np.int8), (1, 1)))
        s = np.flatnonzero(d == 1)
        e = np.flatnonzero(d == -1) - 1
        good = (e - s + 1) >= 2
        if not good.any():
            return -1

        inc = np.zeros(n + 1, np.int16)
        np.add.at(inc, s[good],  1)
        np.add.at(inc, e[good] + 1, -1)
        core = np.cumsum(inc[:-1]) > 0

        FWD_EXPAND = 4   # tolerance *after* the zero-run
        BWD_EXPAND = 1   # small tolerance *before* the zero-run

        mask = core.copy()
        # forward dilation
        for k in range(1, FWD_EXPAND + 1):
            mask[k:] |= core[:-k]
        # backward dilation
        for k in range(1, BWD_EXPAND + 1):
            mask[:-k] |= core[k:]

        # # score (0,0) down‑chirp pairs inside mask
        # pair = mask[:-1] & mask[1:] & (down_syms[:-1] == 0) & (down_syms[1:] == 0)
        # if not pair.any():
        #     return -1

        valid_syms = {0, 1, self.phy_params.chips_per_symbol - 1} #Allows a wiggle room for the downchirp pair
        pair = (
            mask[:-1] & mask[1:] &
            np.isin(down_syms[:-1], list(valid_syms)) &
            np.isin(down_syms[1:], list(valid_syms))
        )

        scores = np.where(pair, 0.5 * (down_peaks[:-1] + down_peaks[1:]), -np.inf)
        i = int(np.argmax(scores))
        if scores[i] == -np.inf:
            return -1

        if self.logging:
            print(f"[check_sfd] (0,0) pair at symbol indices {i*self.phy_params.samples_per_symbol}&{(i+1) * self.phy_params.samples_per_symbol}, score={scores[i]:.3f}")

        return int((i + 2 + 0.25) * self.phy_params.samples_per_symbol)

    def _find_sfd_end_offset(self, synced: Union[np.ndarray, "cp.ndarray"]) -> int:
        if len(synced) % self.phy_params.samples_per_symbol != 0:
            synced = synced[:-(len(synced) % self.phy_params.samples_per_symbol)]

        up_syms, up_peaks = self.demod.demodulate(synced, base="downchirp", return_items=["symbols", "peaks"])
        down_syms, down_peaks = self.demod.demodulate(synced, base="upchirp", return_items=["symbols", "peaks"])
        self.demod.backend.clear_memory()

        sfd_end_offset = self._check_sfd(
            up_syms, up_peaks, down_syms, down_peaks
        )
        if sfd_end_offset == -1:
            raise SFDError("Failed to locate the downchirp pair in SFD.")
        self._log("run", f"Downchirp pair found up to index {sfd_end_offset - 1}")
        return sfd_end_offset


    @staticmethod
    def _decode_payload_length(hi: int, lo: int, sf: int) -> int:
        """Decode the payload length from the header symbols."""
        max_symbol = (1 << sf) - 1
        if not (0 <= hi <= max_symbol) or not (0 <= lo <= max_symbol):
            raise ValueError("Header symbols out of range.")
        return (hi << sf) | lo

    def run(self,
        iq_samples: Union[np.ndarray, "cp.ndarray"],
        *,
        viz_bundle: bool = False  # return viz dict when True
        ) -> Union[
            np.ndarray,
            Tuple[np.ndarray, dict],
            Tuple[np.ndarray, List[CandidateTrace]],
            Tuple[np.ndarray, dict, List[CandidateTrace]],
        ]:

        traces: List[CandidateTrace] = []
        candidates = self._find_preamble_candidates(iq_samples)

        self._log("run", f"Found {len(candidates)} candidates for synchronization.")
        
        if self.logging:
            for cand in candidates:
                self._log("run", f"Candidate: {str(cand)}")

        for i, cand in enumerate(candidates):
            trace = CandidateTrace(index=i, candidate=cand)
            try:
                self._log("run", f"Processing candidate {i+1}/{len(candidates)}")

                # 1. Sync Offset calculation
                sync_offset = self._sync_offset_from_candidate(cand)
                trace.sync_offset = sync_offset

                # 2) Align and optionally CFO/SFO-correct
                synced = iq_samples[sync_offset:]
                synced = self._estimate_and_maybe_apply_cfo_sfo(synced, trace)

                # 3) SFD end offset
                sfd_end_offset = self._find_sfd_end_offset(synced)
                trace.sfd_end_offset = sfd_end_offset

                # 4) Body extraction
                body = iq_samples[sync_offset + sfd_end_offset:]
                if len(body) % self.phy_params.samples_per_symbol != 0:
                    body = body[:-(len(body) % self.phy_params.samples_per_symbol)]

                # 5) Header extraction (2 symbols)
                sps = self.phy_params.samples_per_symbol
                header = body[:2 * sps]
                if len(header) < 2 * sps:
                    raise IncompleteHeaderError("IQ buffer too short to extract the header.")
                
                # 6) Header demodulation: Payload length decoding
                self._log("run", f"Header length: {len(header)} samples. Starts at {sfd_end_offset} samples from sync_offset {sync_offset}.")
                header_syms = self.demod.demodulate(header, base="downchirp", return_items=["symbols"])
                hi, lo = int(header_syms[0]), int(header_syms[1])
                trace.header_syms = (hi, lo)

                payload_len = self._decode_payload_length(hi, lo, self.phy_params.spreading_factor)
                trace.decoded_payload_len = payload_len
                self._log("run", f"Decoded payload length: {payload_len} symbols from({header_syms})")

                needed_samples = int(payload_len * sps)
                available_samples = len(body) - 2 * sps
                trace.needed_samples = needed_samples
                trace.available_samples = available_samples
                self._log("run", f"Needed {needed_samples} samples, available {available_samples} samples in the body.")
                if needed_samples > available_samples:
                    raise IncompletePayloadError("IQ buffer too short to extract as many payload symbols as the header indicates.")

                # 7) Payload extraction
                frame_samples = self.frame_params.HEADER_LENGTH * sps
                payload_frame = body[frame_samples: frame_samples + needed_samples]

                if viz_bundle:
                    header_start_offset   = sync_offset + sfd_end_offset
                    payload_start_offset  = header_start_offset + 2 * sps
                    payload_end_offset    = payload_start_offset + needed_samples
                    sfd_start_offset      = header_start_offset - int(2.25 * sps)
                    sync_word_start_offset = sfd_start_offset - 2 * sps
                    preamble_start_offset  = max(sync_word_start_offset - self.frame_params.preamble_symbol_count * sps, 0)

                    self._log("run", f"Returned offsets: Preamble {preamble_start_offset}, "
                                      f"Sync Word: {sync_word_start_offset}, SFD: {sfd_start_offset}, "
                                      f"Header: {header_start_offset}, Payload: {payload_start_offset}")

                    viz_dict = {
                        "phy_params": self.phy_params,
                        "iq_samples": iq_samples,  # reference, no copy
                        "preamble_start_offset": preamble_start_offset,
                        "sync_word_start_offset": sync_word_start_offset,
                        "sfd_start_offset": sfd_start_offset,
                        "header_start_offset": header_start_offset,
                        "payload_start_offset": payload_start_offset,
                        "payload_end_offset": payload_end_offset,
                    }

                # Success & return according to flags
                trace.status = "success"
                if self.debug:
                    traces.append(trace)

                if viz_bundle and self.debug:
                    return payload_frame, viz_dict, traces
                if viz_bundle:
                    return payload_frame, viz_dict
                if self.debug:
                    return payload_frame, traces
                return payload_frame

            except Exception as e:
                # Record failure
                trace.status = "error"
                trace.error_type = type(e).__name__
                trace.error_msg = str(e)
                if self.debug:
                    traces.append(trace)

                self._log("run", f"Candidate {i + 1} failed: {e}")

                continue

        # All candidates failed
        exc = CandidatesExhaustedError("All candidates were exhausted without successful synchronization.")
        if self.debug:
            setattr(exc, "traces", traces)  # attach for post-mortem use
        raise exc

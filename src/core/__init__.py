"""
LoRa modem core
===============

This package provides the foundational components used across modulation,
demodulation, and synchronization subsystems.

"""

from .backend import choose_backend

from .params import LoRaPhyParams, LoRaFrameParams
from .markers import LoRaMarkers

from .primitives import (
    generate_base_chirp,
    instantaneous_phase,
    to_complex,
)

from .sdr_utils import (
    delete_sdr,
    soft_delete_sdr,
    optimize_payload_symbols,
)

from .misc_utils import (
    maybe_dump_iq_buffer,
    load_iq_dump,
    list_iq_dumps,
    IQDump,
)

from .snr_utils import estimate_snr_from_ls_fit

__all__ = [
    # Backend
    "choose_backend",

    # Configuration
    "LoRaPhyParams",
    "LoRaFrameParams",
    "LoRaMarkers",

    # DSP helpers
    "generate_base_chirp",
    "instantaneous_phase",
    "to_complex",

    # SDR utils
    "delete_sdr",
    "soft_delete_sdr",
    "optimize_payload_symbols",

    # IQ dump utils
    "maybe_dump_iq_buffer",
    "load_iq_dump",
    "list_iq_dumps",
    "IQDump",

    # SNR estimation
    "estimate_snr_from_ls_fit",
]

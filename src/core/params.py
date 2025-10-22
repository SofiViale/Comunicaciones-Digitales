"""
LoRa Physical Layer Parameters
========================

Immutable physical-layer parameters shared by TX/RX blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from warnings import warn

__all__ = ["LoRaParams"]


@dataclass(slots=True) # Inmutable and memory-efficient
class LoRaPhyParams:
    """
    LoRa physical-layer parameters
    
    :param spreading_factor: Spreading factor (SF) 7 … 12.
    :type  spreading_factor: int
    :param bandwidth: Bandwidth in Hz (125e3, 250e3, 500e3).
    :type  bandwidth: float
    :param samples_per_chip: Number of discrete-time samples per chip (SPC > 0).
    :type  samples_per_chip: int
    """

    spreading_factor: int  # 7 … 12
    bandwidth: float       # 125e3, 250e3, 500e3 (Hz)
    samples_per_chip: int  # > 0
    # ---------------------- Validations ------------------------------
    def __post_init__(self) -> None:  # noqa: D401
        if self.spreading_factor not in range(7, 13):
            raise ValueError("spreading_factor must be 7 … 12")
        if self.bandwidth not in (125e3, 250e3, 500e3):
            print("[WARN - PHY LoRA params] standard bandwidths are 125 kHz, 250 kHz or 500 kHz")
        if self.samples_per_chip <= 0:
            raise ValueError("samples_per_chip must be positive")
        if self.samples_per_chip > 2:
            print(
                "[WARN - PHY LoRA params] samples_per_chip > 2 is not standard. "
                "This may lead to unexpected results in some cases. (FPA and CPA folding may not work as expected)",
            )

    # ---------------- Convenience Derived Properties -----------------------------
    @property
    def chips_per_symbol(self) -> int:
        """2**SF."""
        return 1 << self.spreading_factor

    @property
    def samples_per_symbol(self) -> int:
        """Discrete-time samples per LoRa symbol."""
        return self.chips_per_symbol * self.samples_per_chip

    @property
    def symbol_duration(self) -> float:
        """Symbol period *T_sym* in seconds [s]."""
        return self.chips_per_symbol / self.bandwidth

    @property
    def sample_duration(self) -> float:
        """Sample period *T_s* in seconds [s]."""
        return self.symbol_duration / self.samples_per_symbol

    @property
    def sample_rate(self) -> float:
        """Sample rate *f_s* in samples per second [Hz]."""
        return self.samples_per_chip * self.bandwidth

    # ---------------- Repr ------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        return (
            f"LoRaPhyParams(SF={self.spreading_factor}, BW={self.bandwidth/1e3:.0f} kHz, "
            f"SPC={self.samples_per_chip})"
        )
    
    def to_dict(self) -> dict:
        """Return a dictionary representation of the parameters."""
        return {
            "spreading_factor": self.spreading_factor,
            "bandwidth": self.bandwidth,
            "samples_per_chip": self.samples_per_chip
        }
    
    @staticmethod
    def from_dict(data: dict) -> LoRaPhyParams:
        """Create a LoRaPhyParams instance from a dictionary."""
        return LoRaPhyParams(
            spreading_factor=data["spreading_factor"],
            bandwidth=data["bandwidth"],
            samples_per_chip=data["samples_per_chip"]
        )


@dataclass(slots=True)
class LoRaFrameParams:
    """
    LoRa frame parameters.

    :param preamble_symbol_count: Number of up-chirps in the preamble (≥0, recommended ≥8).
    :type  preamble_symbol_count: int
    :param explicit_header: If *True*, a two-symbol header with payload length is sent.
    :type  explicit_header: bool
    :param sync_word: 1-byte network sync word (0-255). Use 0x00 as it is not yet implemented.
    :type  sync_word: int
    :param SYNC_WORD_LENGTH: Number of symbols in the sync word (default 2).
    :type  SYNC_WORD_LENGTH: int
    :param SFD_LENGTH: Length of the SFD in symbols (default 2.25).
    :type  SFD_LENGTH: float
    :param HEADER_LENGTH: Length of the header in symbols (default 2).
    :type  HEADER_LENGTH: int
    :param SYNC_SECTION_LENGTH: Total length of the sync section in symbols (default preamble_symbol_count + SYNC_WORD_LENGTH + SFD_LENGTH + HEADER_LENGTH).
    :type  SYNC_SECTION_LENGTH: float
    """

    preamble_symbol_count: int = 8
    explicit_header: bool = True 
    sync_word: int = 0x00  
    SYNC_WORD_LENGTH: int = 2 
    SFD_LENGTH: float = 2.25
    HEADER_LENGTH: int = 2
    SYNC_SECTION_LENGTH: float = field(init=False)

    # ---------------------- Validations ------------------------------
    def __post_init__(self) -> None:  # noqa: D401
        if self.preamble_symbol_count < 0:
            raise ValueError("preamble_symbol_count must be >= 0")
        if not (0 <= self.sync_word <= 255):
            raise ValueError("sync_word must be in range 0 … 255")
        if not isinstance(self.explicit_header, bool):
            raise TypeError("explicit_header must be a boolean")
        if not isinstance(self.sync_word, int):
            raise TypeError("sync_word must be an integer")
        if self.sync_word != 0x00:
            raise ValueError(
                "NotYetImplementedError: sync_word != 0x00 is not yet implemented. Use 0x00 for now."
            )
        self.SYNC_SECTION_LENGTH = (
            self.preamble_symbol_count + self.SYNC_WORD_LENGTH + self.SFD_LENGTH + self.HEADER_LENGTH
        )

    # ---------------- Convenience Derived Properties -----------------------------
    def encode_sync_word(self, bits_per_symbol: int) -> list[int]:
        """Encode the sync word into a list of two symbols."""
        if bits_per_symbol <= 0:
            raise ValueError("bits_per_symbol must be positive")

        total_bits = 2 * bits_per_symbol
        binary_str = f"{self.sync_word:0{total_bits}b}"  # zero-pad to total_bits
        return [
            int(binary_str[0:bits_per_symbol], 2),       # First symbol     
            int(binary_str[bits_per_symbol:], 2),        # Second symbol    
        ]

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"LoRaFrameParams(preamble={self.preamble_symbol_count}, "
            f"explicit_header={self.explicit_header}, sync_word=0x{self.sync_word:02X}, "
            f"sync_section_length={self.SYNC_SECTION_LENGTH:.2f} symbols)"
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the frame parameters."""
        return {
            "preamble_symbol_count": self.preamble_symbol_count,
            "explicit_header": self.explicit_header,
            "sync_word": self.sync_word,
            "SYNC_WORD_LENGTH": self.SYNC_WORD_LENGTH,
            "SFD_LENGTH": self.SFD_LENGTH,
            "HEADER_LENGTH": self.HEADER_LENGTH,
            "SYNC_SECTION_LENGTH": self.SYNC_SECTION_LENGTH
        }

    @staticmethod
    def from_dict(data: dict) -> LoRaFrameParams:
        """Create a LoRaFrameParams instance from a dictionary."""
        return LoRaFrameParams(
            preamble_symbol_count=data["preamble_symbol_count"],
            explicit_header=data["explicit_header"],
            sync_word=data["sync_word"],
            SYNC_WORD_LENGTH=data.get("SYNC_WORD_LENGTH", 2),
            SFD_LENGTH=data.get("SFD_LENGTH", 2.25),
            HEADER_LENGTH=data.get("HEADER_LENGTH", 2)
        )

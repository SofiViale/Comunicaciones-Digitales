"""
LoRa Markers
================

LoRa modulation markers used in the PHY layer.
"""

from enum import Enum

class LoRaMarkers(Enum):
    """
    Reserved LoRa modulation markers used in the PHY layer.

    These are not data-carrying symbols but fixed waveform patterns used
    for synchronization and framing purposes (e.g., preamble and SFD).

    :cvar FULL_UPCHIRP: Full-duration upchirp (identical to symbol 0).  
                        Used in the preamble.
    :cvar FULL_DOWNCHIRP: Full-duration downchirp (symbol 0, reversed slope).  
                          Used in the start frame delimiter (SFD).
    :cvar QUARTER_DOWNCHIRP: Quarter-duration downchirp (symbol 0, reversed slope, 1/4 time).  
                             Marks the end of the SFD.
    """

    FULL_UPCHIRP = (0, 1.0, 1)
    FULL_DOWNCHIRP = (0, 1.0, -1)
    QUARTER_DOWNCHIRP = (0, 0.25, -1)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def symbol_val(self) -> int:
        """Symbol value to modulate (always 0)."""
        return self.value[0]

    @property
    def duration_factor(self) -> float:
        """Relative duration factor (1.0 = full symbol, 0.25 = quarter)."""
        return self.value[1]

    @property
    def slope_sign(self) -> int:
        """+1 for upchirp, -1 for downchirp."""
        return self.value[2]

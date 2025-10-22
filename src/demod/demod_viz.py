# demod_viz.py
from __future__ import annotations
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

_SYMBOL_COLORS = plt.cm.get_cmap('hsv', 256)

def _get_color(symbol: int, cmap_name: str = "Set1") -> tuple:
    """
    Return a pastel RGBA color for a given integer symbol.
    """
    import hashlib, matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)          # 'Pastel1' or 'Pastel2'
    # deterministic hash → float in [0,1)
    h = int(hashlib.sha256(str(symbol).encode()).hexdigest()[:8], 16)
    return cmap((h % 10_000) / 10_000)



def plot_demodulation(debug_bundle: Dict[str, Any]) -> None:
    phy         = debug_bundle["phy_params"]
    tx_waveform = np.asarray(debug_bundle["modulated_symbols"])
    symbols     = np.asarray(debug_bundle["demodulated_symbols"])
    peaks       = np.asarray(debug_bundle["peak_magnitudes"])
    folded_mag  = np.asarray(debug_bundle["folded_mag_fft"])

    sps = phy.samples_per_symbol
    num_symbols = len(symbols)

    _, unique_indices = np.unique(symbols, return_index=True)
    unique_symbols = symbols[np.sort(unique_indices)]
    symbol_to_color = {sym: _get_color(sym) for sym in unique_symbols}

    # Create full figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("LoRa Demodulation Visualization", fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 2, 2, 2])

    # 1. Legend
    ax_legend = fig.add_subplot(gs[0])
    legend_lines = []
    legend_labels = []

    MAX_LEGEND = 30  # Para no saturar

    for sym in unique_symbols[:MAX_LEGEND]:
        color = symbol_to_color[sym]
        line, = ax_legend.plot([], [], marker='s', linestyle='None', color=color, label=f"{sym}")
        legend_lines.append(line)
        legend_labels.append(f"{sym}")

    ax_legend.legend(handles=legend_lines, labels=legend_labels, loc="center", ncol=min(MAX_LEGEND, 10), frameon=False)
    ax_legend.axis("off")
    ax_legend.set_title("Symbol → Color Mapping")
    # 2. I and Q signal segments per symbol
    ax_real = fig.add_subplot(gs[1])
    ax_imag = fig.add_subplot(gs[2], sharex=ax_real)

    for i in range(num_symbols):
        sym = symbols[i]
        color = symbol_to_color[sym]
        segment = tx_waveform[i * sps:(i + 1) * sps]
        time_axis = np.arange(i * sps, (i + 1) * sps)

        ax_real.plot(time_axis, segment.real, color=color)
        ax_imag.plot(time_axis, segment.imag, color=color)

    ax_real.set_title("I (Real Part) per Symbol")
    ax_imag.set_title("Q (Imag Part) per Symbol")
    ax_imag.set_xlabel("Sample Index")
    ax_real.grid(alpha=0.3)
    ax_imag.grid(alpha=0.3)

    # 3. Folded FFT Magnitude
    ax_fft = fig.add_subplot(gs[3])

    # Ensure folded_mag is 2D
    if folded_mag.ndim == 1:
        folded_mag = np.expand_dims(folded_mag, axis=0)

    for i in range(num_symbols):
        sym = symbols[i]
        color = symbol_to_color[sym]
        spectrum = folded_mag[i]

        if not hasattr(spectrum, "__len__") or sym >= len(spectrum):
            continue  # Skip invalid entries

        ax_fft.plot(spectrum[:debug_bundle["phy_params"].chips_per_symbol], color=color, alpha=0.6)

        # Annotate peak with bold text
        peak_idx = sym
        peak_val = spectrum[peak_idx]


    ax_fft.set_title("Folded FFT Magnitude per Symbol")
    ax_fft.set_xlabel("FFT Bin (Symbol Index)")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

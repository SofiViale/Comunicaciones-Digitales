# demod_viz.py
from __future__ import annotations
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np

def _get_color(symbol: float, cmap_name: str = "Set1") -> tuple:
    """
    Return a pastel RGBA color for a given symbol.
    """
    import hashlib, matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    # deterministic hash → float in [0,1)
    h = int(hashlib.sha256(str(symbol).encode()).hexdigest()[:8], 16)
    return cmap((h % 10_000) / 10_000)


def plot_demodulation(debug_bundle: Dict[str, Any], *, title="LoRa Demodulation Visualization", dft_color=None) -> None:
    phy         = debug_bundle["phy_params"]
    tx_waveform = np.asarray(debug_bundle["modulated_symbols"])
    symbols     = np.asarray(debug_bundle["demodulated_symbols"])
    peaks       = np.asarray(debug_bundle["peak_magnitudes"])
    folded_mag  = np.asarray(debug_bundle["folded_mag_fft"])
    padding     = debug_bundle.get("padding", 1)

    sps = phy.samples_per_symbol
    num_symbols = len(symbols)
    
    # Calculate chips based on SF assuming standard LoRa (2^SF)
    chips = 1 << phy.spreading_factor

    _, unique_indices = np.unique(symbols, return_index=True)
    unique_symbols = symbols[np.sort(unique_indices)]
    symbol_to_color = {sym: _get_color(sym) for sym in unique_symbols}

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 2, 2, 2])

    # 1. Legend
    ax_legend = fig.add_subplot(gs[0])
    legend_lines = []
    legend_labels = []

    MAX_LEGEND = 30

    for sym in unique_symbols[:MAX_LEGEND]:
        if dft_color is None:
            color = symbol_to_color[sym]
        else:
            color = dft_color
        
        # Display up to 2 decimal places for float symbols
        label_text = f"{sym:.2f}"
        line, = ax_legend.plot([], [], marker='s', linestyle='None', color=color, label=label_text)
        legend_lines.append(line)
        legend_labels.append(label_text)

    ax_legend.legend(handles=legend_lines, labels=legend_labels, loc="center", ncol=min(MAX_LEGEND, 10), frameon=False)
    ax_legend.axis("off")
    ax_legend.set_title("Symbol → Color Mapping")

    # 2. I and Q signal segments per symbol
    ax_real = fig.add_subplot(gs[1])
    ax_imag = fig.add_subplot(gs[2], sharex=ax_real)

    for i in range(num_symbols):
        sym = symbols[i]
        if dft_color is None:
            color = symbol_to_color[sym]
        else:
            color = dft_color
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

    if folded_mag.ndim == 1:
        folded_mag = np.expand_dims(folded_mag, axis=0)

    for i in range(num_symbols):
        sym = symbols[i]
        if dft_color is None:
            color = symbol_to_color[sym]
        else:
            color = dft_color
        
        spectrum = folded_mag[i]
        
        if not hasattr(spectrum, "__len__"):
            continue

        # Generate X-axis scaled to Symbol Values (0 to 2^SF)
        # The spectrum length is chips * padding.
        fft_bins = len(spectrum)
        x_axis = np.linspace(0, chips, fft_bins, endpoint=False)

        ax_fft.plot(x_axis, spectrum, color=color, alpha=0.6)

        # Calculate the index in the zero-padded array corresponding to the detected symbol
        peak_bin_idx = int(round(sym * padding))
        
        if 0 <= peak_bin_idx < fft_bins:
            peak_val = spectrum[peak_bin_idx]
            # Plot a marker at the detected peak
            ax_fft.plot(sym, peak_val, marker='x', color='black', markersize=8, markeredgewidth=2)
            
            # Annotate only if sparsely populated or specific index
            if i % max(1, num_symbols // 10) == 0:
                ax_fft.annotate(f"{sym:.2f}", (sym, peak_val), 
                                textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    ax_fft.set_title(f"Folded FFT Magnitude")
    ax_fft.set_xlabel("Symbol Value")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.set_xlim(0, chips)
    ax_fft.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

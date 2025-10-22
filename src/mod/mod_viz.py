# mod/viz.py
from __future__ import annotations
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import hashlib

_COLORS = {
    "Preamble": "#F4C430",  
    "SyncWord": "#66BB6A",  
    "SFD":      "#29B6F6",  
    "Header":   "#5C6BC0",  
    "Payload":  "#AB47BC",  
}


def _get_color(symbol: int, cmap_name: str = "Set1") -> tuple:
    """
    Return a pastel RGBA color for a given integer symbol.
    """
    import hashlib, matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)          # 'Pastel1' or 'Pastel2'
    # deterministic hash → float in [0,1)
    h = int(hashlib.sha256(str(symbol).encode()).hexdigest()[:8], 16)
    return cmap((h % 10_000) / 10_000)

from matplotlib.gridspec import GridSpec

def plot_frame(bundle: Dict[str, Any]) -> None:
    """
    Visualise either a full LoRa *frame* or a *payload-only* waveform
    produced by ``LoRaModulator.modulate(debug_bundle=True)``.
    """
    t     = np.asarray(bundle["time_axis"])
    freq  = np.asarray(bundle["instantaneous_frequency"])
    sig   = np.asarray(bundle["signal"])

    mode  = bundle.get("mode", "payload_only")  # "frame" | "payload_only"
    segments: Dict[str, tuple[int, int]]

    symbols = bundle["payload_symbols"]
    sps     = bundle["samples_per_chip"] * (1 << bundle["spreading_factor"])

    # --- Setup figure with optional legend row ---
    if mode == "payload_only":
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 1, height_ratios=[0.5, 2, 2, 2])
        ax_legend = fig.add_subplot(gs[0])
        axs = [fig.add_subplot(gs[i]) for i in range(1, 4)]
    else:
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))

    title = "LoRa Modulation Visualization: FRAME" if mode == "frame" else "LoRa Modulation Visualization: PAYLOAD-only"

    fig.suptitle(title, fontsize=16, fontweight='bold')

    if mode == "frame":
        idx     = bundle["indexes"]
        sf      = bundle["spreading_factor"]
        sps     = bundle["samples_per_chip"] * (1 << sf)
        has_hdr = bundle.get("has_explicit_header", False)

        sync_end = idx["preamble_end"] + 2 * sps
        segments = {
            "Preamble": (0, idx["preamble_end"]),
            "SyncWord": (idx["preamble_end"] + 1, sync_end),
            "SFD":      (sync_end + 1, idx["sfd_end"]),
        }
        if has_hdr and idx["header_end"] is not None:
            segments["Header"]  = (idx["sfd_end"] + 1, idx["header_end"])
            segments["Payload"] = (idx["header_end"] + 1, len(sig) - 1)
        else:
            segments["Payload"] = (idx["sfd_end"] + 1, len(sig) - 1)

        for name, (lo, hi) in segments.items():
            axs[0].plot(t[lo:hi + 1], freq[lo:hi + 1], color=_COLORS[name], label=name, lw=1.5)
            axs[1].plot(t[lo:hi + 1], sig.real[lo:hi + 1], color=_COLORS[name], lw=1)
            axs[2].plot(t[lo:hi + 1], sig.imag[lo:hi + 1], color=_COLORS[name], lw=1)
        axs[0].legend(loc="upper right")

    else:  # payload-only visualización por símbolo
        MAX_LEGEND = 20
        legend_lines = []
        legend_labels = []
        seen_symbols = set()

        _, unique_indices = np.unique(symbols, return_index=True)
        symbols_np = np.asarray(symbols)
        ordered_unique_symbols = symbols_np[np.sort(unique_indices)]


        for i, sym in enumerate(symbols):
            lo = i * sps
            hi = (i + 1) * sps
            color = _get_color(sym)

            axs[0].plot(t[lo:hi], freq[lo:hi], color=color, lw=1.5)
            axs[1].plot(t[lo:hi], sig.real[lo:hi], color=color, lw=1)
            axs[2].plot(t[lo:hi], sig.imag[lo:hi], color=color, lw=1)

        # Ahora generamos la leyenda de forma separada y sin duplicados
        for sym in ordered_unique_symbols[:MAX_LEGEND]:
            color = _get_color(sym)
            line, = ax_legend.plot([], [], marker='s', linestyle='None', color=color, label=f"{sym}")
            legend_lines.append(line)
            legend_labels.append(f"{sym}")

        ax_legend.legend(handles=legend_lines, labels=legend_labels, loc="center", ncol=MAX_LEGEND, frameon=False)
        ax_legend.axis("off")
        ax_legend.set_title("Symbol → Color Mapping")


    axs[0].set(title="Instantaneous Frequency", ylabel="Hz")
    axs[1].set(title="I (Real Part) per Symbol")
    axs[2].set(title="Q (Imag Part) per Symbol", xlabel="Time [s]")

    for ax in axs:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

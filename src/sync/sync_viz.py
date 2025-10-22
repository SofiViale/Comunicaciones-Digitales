import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

DEFAULT_PAD_SAMPLES = 100

_COLORS = {
    "Preamble": "#F4C430",  
    "SyncWord": "#66BB6A",  
    "SFD":      "#29B6F6",  
    "Header":   "#5C6BC0",  
    "Payload":  "#AB47BC",  
    "Padding":  "#444444",
}

def plot_synchronization(debug_bundle: Dict[str, Any], *, pad_samples:int = None) -> None:
    phy = debug_bundle["phy_params"]
    iq = np.asarray(debug_bundle["iq_samples"])
    sps = phy.samples_per_symbol


    # Compute absolute segment boundares (in samples)
    pre_start = debug_bundle["preamble_start_offset"]
    sync_start = debug_bundle["sync_word_start_offset"]
    sfd_start = debug_bundle["sfd_start_offset"]
    hdr_start = debug_bundle["header_start_offset"]
    pl_start = debug_bundle["payload_start_offset"]
    pl_end = debug_bundle["payload_end_offset"]

    # Add padding around the entire frame window
    pad = pad_samples if pad_samples else DEFAULT_PAD_SAMPLES
    vis_start = max(pre_start - pad, 0)
    vis_end = min(pl_end + pad, len(iq))

    time_axis = np.arange(vis_start, vis_end)
    segment_iq = iq[vis_start:vis_end]

    # Relative positions within the plot window
    rel = lambda x: x - vis_start

    # Define segment boundaries for coloring
    segments = [
        ("Padding",  vis_start, pre_start),
        ("Preamble", pre_start, sync_start),
        ("SyncWord", sync_start, sfd_start),
        ("SFD",     sfd_start, hdr_start),
        ("Header",   hdr_start, pl_start),
        ("Payload",  pl_start,  pl_end),
        ("Padding",  pl_end,    vis_end),
    ]

    fig, (ax_real, ax_imag) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    fig.suptitle("LoRa Dechirp-based Synchronization: Frame View", fontsize=16, fontweight="bold")

    for name, start, end in segments:
        color = _COLORS[name]
        r_start, r_end = rel(start), rel(end)
        if r_start >= r_end:
            continue
        seg_axis = time_axis[r_start:r_end]
        seg_iq = segment_iq[r_start:r_end]
        ax_real.plot(seg_axis, seg_iq.real, color=color, linewidth=0.8, label=name if name != "Padding" else None)
        ax_imag.plot(seg_axis, seg_iq.imag, color=color, linewidth=0.8)

    for ax in [ax_real, ax_imag]:
        ax.grid(alpha=0.3)


    ax_real.set_title("I (Real Part)")
    ax_imag.set_title("Q (Imag Part)")
    ax_imag.set_xlabel("Sample Index")

    # Construct legend manually (ignore repeated entries)
    handles, labels = ax_real.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_real.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    plt.show()

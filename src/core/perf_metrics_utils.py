from dataclasses import dataclass
from src.core.snr_utils import SDRProfile
from src.core.misc_utils import get_class_persistence_manager
from pathlib import Path
from typing import List, Tuple, Optional
from statistics import NormalDist
from math import sqrt

# ----------------------------------------------------------------------
# Wilson CI utility
# ----------------------------------------------------------------------
def wilson_ci(event_count: int,
              total_trial_count: int,
              conf_level: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion (event_count / total_trial_count).

    Returns (ci_low, ci_high).
    """
    if total_trial_count <= 0:
        raise ValueError("total_trial_count must be > 0 for Wilson CI.")

    # z for two-sided confidence
    z = NormalDist().inv_cdf(1 - (1 - conf_level) / 2)

    p_hat = event_count / total_trial_count
    denom = 1.0 + (z * z) / total_trial_count
    center = (p_hat + (z * z) / (2 * total_trial_count)) / denom
    margin = (z * sqrt(
        (p_hat * (1 - p_hat) / total_trial_count) +
        (z * z) / (4 * total_trial_count * total_trial_count)
    )) / denom

    ci_low = max(0.0, center - margin)
    ci_high = min(1.0, center + margin)
    return ci_low, ci_high

@dataclass
class SinglePerformancePoint:
    """
    Binomial rate estimate (p_hat) and confidence interval for a single performance metric.
    
    May represent Bit Error Rate (BER), Symbol Error Rate (SER), or Frame Error Rate (FER).
    """
    event_count: int         # Observed events (Errors, for example)
    total_trial_count: int   # Total trials (Symbols, Frames, etc.)
    p_hat: float             # Estimated probability of the event occurring (event_count / total_trial_count)
    ci_low: float            # Lower bound of the confidence interval
    ci_high: float           # Upper bound of the confidence interval
    rel_half_width: float    # Relative half-width of the confidence interval ( ((ci_high-ci_low)/2) / max(p_hat, 1/total) )
    conf_level: float = 0.95 # Confidence level used for the interval (e.g., 0.95 for 95% confidence)

    @staticmethod
    def from_event_count(event_count: int, total_trial_count: int, ci_fn=wilson_ci, conf_level: float = 0.95) -> 'SinglePerformancePoint':
        if total_trial_count == 0:
            raise ValueError("Total trial count cannot be zero.")
        
        p_hat = event_count / total_trial_count

        ci_low, ci_high = ci_fn(event_count, total_trial_count, conf_level)

        half_width = (ci_high - ci_low) / 2
        rel_half_width = half_width / max(p_hat, 1 / total_trial_count)

        return SinglePerformancePoint(
            event_count=event_count,
            total_trial_count=total_trial_count,
            p_hat=p_hat,
            ci_low=ci_low,
            ci_high=ci_high,
            rel_half_width=rel_half_width,
            conf_level=conf_level
        )

    def is_trusted(self, rel_tolerance: float) -> bool:
        """
        Heuristic to determine if the performance point is statistically reliable.
        
        Uses the relative half-width of the confidence interval.
        """
        return self.rel_half_width <= rel_tolerance

    @property
    def abs_half_width(self) -> float:
        return (self.ci_high - self.ci_low)/2
    
    def to_dict(self) -> dict:
        """Convert the performance point to a dictionary."""
        return {
            "event_count": int(self.event_count),
            "total_trial_count": int(self.total_trial_count),
            "p_hat": float(self.p_hat),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "rel_half_width": float(self.rel_half_width),
            "conf_level": float(self.conf_level)
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'SinglePerformancePoint':
        """Create a SinglePerformancePoint instance from a dictionary."""
        return SinglePerformancePoint(
            event_count=data["event_count"],
            total_trial_count=data["total_trial_count"],
            p_hat=data["p_hat"],
            ci_low=data["ci_low"],
            ci_high=data["ci_high"],
            rel_half_width=data["rel_half_width"],
            conf_level=data["conf_level"]
        )

@dataclass
class SNRPointPerformanceMetrics:
    """
    Performance metrics for a single SNR point.
    
    Holds a given SNR value and the performance metrics for BER, SER, and FER.
    """
    snr_db: float
    ber: Optional[SinglePerformancePoint] = None
    ser: Optional[SinglePerformancePoint] = None
    fer: Optional[SinglePerformancePoint] = None

    def to_dict(self) -> dict:
        """Convert the SNR point performance metrics to a dictionary."""
        return {
            "snr_db": float(self.snr_db),
            "ber": self.ber.to_dict() if self.ber else None,
            "ser": self.ser.to_dict() if self.ser else None,
            "fer": self.fer.to_dict() if self.fer else None
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'SNRPointPerformanceMetrics':
        """Create a SNRPointPerformanceMetrics instance from a dictionary."""
        return SNRPointPerformanceMetrics(
            snr_db=data["snr_db"],
            ber=SinglePerformancePoint.from_dict(data["ber"]) if data["ber"] else None,
            ser=SinglePerformancePoint.from_dict(data["ser"]) if data["ser"] else None,
            fer=SinglePerformancePoint.from_dict(data["fer"]) if data["fer"] else None
        )

@dataclass
class SNRPerformanceMetrics:
    """
    Performance metrics for multiple SNR points based on a specific SDR profile.

    Holds a list of SNRPointPerformanceMetrics for each SNR point.
    Also includes the SDR profile used for these metrics.
    """
    profile: SDRProfile
    snr_points: list[SNRPointPerformanceMetrics]
    channel_function:str = "unknown"

    def auto_name(self, extension: str = ".json", *, include_name: bool = False) -> str:
        prefix = "perf_"
        stem = self.profile.file_stem(include_name=include_name)
        ext = extension if extension.startswith(".") else f".{extension}"
        return prefix + stem + ext

    def save(self, filename: str | None = None) -> Path:
        pm = get_class_persistence_manager(type(self))
        return pm.save(self, filename=filename, namer=SNRPerformanceMetrics.auto_name)
    
    @classmethod
    def load(cls, filename: str) -> 'SNRPerformanceMetrics':
        pm = get_class_persistence_manager(cls)
        return pm.load(filename, from_dict=cls.from_dict)

    @classmethod
    def list_perf_profiles(cls) -> List[str]:
        """Return available profile performances names without extension."""
        pm = get_class_persistence_manager(cls)
        return pm.list()

    def to_dict(self) -> dict:  
        """Convert the SNR performance metrics to a dictionary."""
        return {
            "profile": self.profile.to_dict(),
            "channel_function": str(self.channel_function),
            "snr_points": [pt.to_dict() for pt in self.snr_points]
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'SNRPerformanceMetrics':
        """Create a SNRPerformanceMetrics instance from a dictionary."""
        profile = SDRProfile.from_dict(data["profile"])
        snr_points = [SNRPointPerformanceMetrics.from_dict(pt) for pt in data["snr_points"]]
        return SNRPerformanceMetrics(
            profile=profile,
            channel_function=data.get("channel_function", "unknown"),
            snr_points=snr_points
        )
    
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def annotate_perf_metadata(snr_metrics, ax: plt.Axes, location='lower left', fontsize=12):
    """
    Annotates the given Axes with key performance metadata extracted from the SNRPerformanceMetrics object.

    - Shows basic PHY/frame parameters
    - Optionally includes SDR Tx/Rx configuration
    - Condenses SNR mapping to a linear regression line
    
    Parameters:
    - snr_metrics: SNRPerformanceMetrics
    - ax: matplotlib Axes
    - location: where to place the annotation ('upper left', 'lower right', etc.)
    - fontsize: text font size
    """
    profile = snr_metrics.profile.to_dict()
    channel_fn = snr_metrics.channel_function

    lines = []

    # Basic
    lines.append(f"Profile: {profile.get('name', '-')}")
    lines.append(f"Channel: {channel_fn}")
    lines.append("")

    # PHY
    phy = profile['phy_params']
    lines.append(f"SF  = {phy['spreading_factor']}")
    lines.append(f"BW  = {phy['bandwidth'] / 1e3:.0f} kHz")
    lines.append(f"SPC = {phy['samples_per_chip']}")
    lines.append(f"Fold= {profile.get('fold_mode', '-')}")
    lines.append("")

    # SDR TX/RX info (if any)
    tx = profile.get("tx_sdr_params")
    rx = profile.get("rx_sdr_params")

    if tx or rx:
        uri = tx.get("uri") if tx else rx.get("uri")
        sr = tx.get("sample_rate") if tx else rx.get("sample_rate")
        lo = tx.get("lo_freq") if tx else rx.get("lo_freq")
        lines.append(f"SDR: {uri}")
        lines.append(f"Rate: {sr/1e6:.2f} MSps")
        lines.append(f"LO: {lo/1e6:.3f} MHz")
        lines.append("")

    # Linear fit for SNR mapping (if exists)
    snr_map = profile.get("snr_map", [])
    if snr_map:
        atten = []
        mean_snr = []
        for point in snr_map:
            if point["snr_values"]:
                atten.append(point["attenuation"])
                mean_snr.append(np.mean(point["snr_values"]))
        if len(atten) > 1:
            from src.core.snr_utils import linear_regression  # or inline it if preferred
            slope, intercept = linear_regression(np.array(atten), np.array(mean_snr))
            lines.append("SNR ≈ m·Atten + b")
            lines.append(f"m = {slope:.3f}, b = {intercept:.2f}")
            lines.append("")

    lines.append(f"Timestamp: {profile.get('timestamp', '-')}")

    # Prepare annotation text
    textstr = "\n".join(lines)

    # Map location → alignment
    loc_map = {
        "upper left": dict(x=0.01, y=0.99, ha='left', va='top'),
        "upper right": dict(x=0.99, y=0.99, ha='right', va='top'),
        "lower left": dict(x=0.01, y=0.01, ha='left', va='bottom'),
        "lower right": dict(x=0.99, y=0.01, ha='right', va='bottom'),
    }
    loc_args = loc_map.get(location, loc_map["lower left"])

    # Draw on Axes
    ax.text(
        loc_args["x"], loc_args["y"],
        textstr,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=loc_args["va"],
        horizontalalignment=loc_args["ha"],
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.85)
    )

def plot_theoric_perf(ax: plt.Axes, fold_mode_to_show:str="FPA"):
    """
    Plot a hardcoded theoretical SER vs SNR curve on the given Axes.
    
    Parameters:
    - ax: matplotlib Axes to draw the plot.
    - fold_mode_to_show: If "CPA", plots CPA curve; if "FPA", plots FPA curve.
    """
    snr_db = np.linspace(-40, 0, 30)

    # Common baseline 
    ser_cpa = 1 / (1 + np.exp((snr_db + 16)/1.5)) 
    ser_fpa = 1 / (1 + np.exp((snr_db + 16.5)/1.5)) 

    if fold_mode_to_show == "CPA":
        ax.plot(snr_db, ser_cpa, ':', label="paper CPA", linewidth=2, color="purple", alpha=0.7)
    elif fold_mode_to_show == "FPA":
        ax.plot(snr_db, ser_fpa, '--', label="paper FPA", linewidth=2, color="purple", alpha=0.7)
    else:
        return

    ax.set_xlim([-40, 0])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SER (%)")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    ax.legend(loc="upper right", fontsize=8)

    ax.text(0.01, 0.99,
            "Theoretical Performance Metric\nin SER - SNR [dB] for SF = 8",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))



def plot_snr_performance(snr_metrics, logscale=True, ax=None):
    """
    Plots performance curves for an SNRPerformanceMetrics object.
    
    :param snr_metrics: SNRPerformanceMetrics object containing the performance data.
    :param logscale: If True, uses a logarithmic scale for the y-axis.
    :param ax: Optional matplotlib Axes object to plot on. If None, creates a new figure and axes.
    :return: The Axes object with the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    snr = np.array([pt.snr_db for pt in snr_metrics.snr_points])

    fer_vals, fer_marks = [], []
    ser_vals, ser_marks = [], []
    ber_vals, ber_marks = [], []

    for pt in snr_metrics.snr_points:
        # FER
        if pt.fer and pt.fer.p_hat is not None:
            fer_vals.append(pt.fer.p_hat)
            fer_marks.append('s' if pt.fer.is_trusted(0.1) else 'x')
        else:
            fer_vals.append(np.nan)
            fer_marks.append('x')

        # SER
        if pt.ser and pt.ser.p_hat is not None:
            ser_vals.append(pt.ser.p_hat)
            ser_marks.append('o' if pt.ser.is_trusted(0.1) else 'x')
        else:
            ser_vals.append(np.nan)
            ser_marks.append('x')

        # BER
        if pt.ber and pt.ber.p_hat is not None:
            ber_vals.append(pt.ber.p_hat)
            ber_marks.append('^' if pt.ber.is_trusted(0.1) else 'x')
        else:
            ber_vals.append(np.nan)
            ber_marks.append('x')

    fer_vals = np.array(fer_vals)
    ser_vals = np.array(ser_vals)
    ber_vals = np.array(ber_vals)

    set_fer_label = set_ser_label = set_ber_label = False

    for i in range(len(snr)):
        if not np.isnan(fer_vals[i]):
            ax.plot(snr[i], fer_vals[i], marker=fer_marks[i], color='C0',
                    label="FER" if not set_fer_label else "")
            set_fer_label = True
        if not np.isnan(ser_vals[i]):
            ax.plot(snr[i], ser_vals[i], marker=ser_marks[i], color='C1',
                    label="SER" if not set_ser_label else "")
            set_ser_label = True
        if not np.isnan(ber_vals[i]):
            ax.plot(snr[i], ber_vals[i], marker=ber_marks[i], color='C2',
                    label="BER" if not set_ber_label else "")
            set_ber_label = True

    # Connect the curves
    ax.plot(snr, fer_vals, "s-", color='C0', alpha=0.4)
    ax.plot(snr, ser_vals, "o-", color='C1', alpha=0.4)
    ax.plot(snr, ber_vals, "^-", color='C2', alpha=0.4)

    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Error Rate (log scale)" if logscale else "Error Rate")
    ax.set_title("LoRa Performance vs SNR")

    if logscale:
        ax.set_yscale("log")

    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend()
    return ax


# ----------------------------------------------------------------------
# Temporal Logs for Performance Metric SNR Points
# ----------------------------------------------------------------------
def create_new_log_file(log_dir:Path, BASE_NAME="last_quant_perf.log") -> Path:
    """
    Create a new log file in the specified directory with an incremental suffix.
    """
    log_dir.mkdir(exist_ok=True)
    base_path = log_dir / BASE_NAME
    if not base_path.exists():
        return base_path
    
    i = 1
    while (log_dir / f"{BASE_NAME.removesuffix('.log')}_{i}.log").exists():
        i += 1
    return log_dir / f"{BASE_NAME.removesuffix('.log')}_{i}.log"

def log_frame_result(log_path: Path, snr_db: float, result: dict, frame_idx: int, syms_total: int):
    """Writes the performance result of a frame to the log file."""
    with open(log_path, "a") as f:
        f.write(f"[SNR {snr_db:+.1f} dB] Frame {frame_idx:06d}: FER={result['fer_err']} SER={result['ser_err']} BER={result['ber_err']} SYMS_TOTAL={syms_total}\n")

from __future__ import annotations
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import time
import json
from typing import Callable, Optional, TypeVar, Any, Type

T = TypeVar("T")
from src.core.params import LoRaPhyParams, LoRaFrameParams

@dataclass
class IQDump:
    iq_buffer: np.ndarray
    payload: list[int]
    phy_params: LoRaPhyParams
    frame_params: LoRaFrameParams


def maybe_dump_iq_buffer(buffer: np.ndarray, payload: list[int], phy_params: LoRaPhyParams,
                          frame_params: LoRaFrameParams, should_dump: bool, caller: str):
    """
    Conditionally dumps an IQ buffer and associated metadata to disk for later debugging.

    Args:
        buffer (np.ndarray): The complex IQ buffer to save.
        payload (list[int]): The original transmitted payload.
        phy_params (LoRaPhyParams): Physical layer parameters.
        frame_params (LoRaFrameParams): Frame structure parameters.
        should_dump (bool): Whether to actually dump the buffer.
        caller (str): Identifier for the calling module/function/test.
    """
    if not should_dump:
        return

    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    debug_dir = project_root / "debug_iq_dumps"
    debug_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"dump_{caller}_{timestamp}.npz"
    full_path = debug_dir / filename

    np.savez_compressed(
        full_path,
        iq_buffer=buffer,
        payload=np.asarray(payload, dtype=np.int32),
        spreading_factor=phy_params.spreading_factor,
        bandwidth=phy_params.bandwidth,
        samples_per_chip=phy_params.samples_per_chip,
        preamble_symbol_count=frame_params.preamble_symbol_count,
        explicit_header=frame_params.explicit_header,
        sync_word=frame_params.sync_word,
    )

    print(f"[DEBUG] IQ buffer dumped to: {full_path}")


def _format_time_ago(file_time: float) -> str:
    now = time.time()
    diff = now - file_time

    if diff < 60:
        return f"{int(diff)} seconds ago"
    elif diff < 3600:
        return f"{int(diff // 60)} minutes ago"
    elif diff < 86400:
        return "at " + time.strftime("%H:%M", time.localtime(file_time))
    else:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_time))


def list_iq_dumps(caller_filter: str | None = None) -> list[Path]:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    debug_dir = project_root / "debug_iq_dumps"
    if not debug_dir.exists():
        print("[INFO] No IQ dump directory found.")
        return []

    files = sorted(debug_dir.glob("*.npz"), key=lambda f: f.stat().st_mtime, reverse=True)
    if caller_filter:
        files = [f for f in files if f.name.startswith(f"dump_{caller_filter}_")]

    for f in files:
        try:
            data = np.load(f)
            sf = data["spreading_factor"]
            bw = data["bandwidth"] / 1e3
            spc = data["samples_per_chip"]
            age = _format_time_ago(f.stat().st_mtime)
            print(f"• {f.name:<50} (SF={sf}, BW={bw:.0f}kHz, SPC={spc}) [{age}]")
        except Exception as e:
            print(f"• {f.name:<50} [Corrupted or invalid file: {e}]")

    return files


def load_iq_dump(filename: str) -> IQDump:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    debug_dir = project_root / "debug_iq_dumps"
    file_path = debug_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"IQ dump file not found: {file_path}")

    data = np.load(file_path)
    required_keys = ["iq_buffer", "payload", "spreading_factor", "bandwidth", "samples_per_chip",
                     "preamble_symbol_count", "explicit_header", "sync_word"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"'{key}' not found in file: {filename}")

    phy_params = LoRaPhyParams(
        spreading_factor=int(data["spreading_factor"]),
        bandwidth=float(data["bandwidth"]),
        samples_per_chip=int(data["samples_per_chip"]),
    )

    frame_params = LoRaFrameParams(
        preamble_symbol_count=int(data["preamble_symbol_count"]),
        explicit_header=bool(data["explicit_header"]),
        sync_word=int(data["sync_word"]),
    )

    payload = data["payload"].tolist()
    iq_buffer = data["iq_buffer"]

    print(f"[DEBUG] Loaded dump: {filename}")
    return IQDump(iq_buffer=iq_buffer, payload=payload,
                  phy_params=phy_params, frame_params=frame_params)

def unsync_frame(frame, offset, max_len=(1 << 20)):
    """Prepends and appends complex noise to misalign the frame."""
    if offset + len(frame) > max_len:
        raise ValueError("Frame too long for the specified offset and max_len.")
    
    return np.concatenate([
        0.015 * (np.random.randn(offset) + 1j * np.random.randn(offset)),
        frame,
        0.015 * (np.random.randn(max_len - len(frame) - offset) + 1j * np.random.randn(max_len - len(frame) - offset))
    ])


#------------------------------------------------------------------------------------
# Persistence utilities
#------------------------------------------------------------------------------------
def find_project_root(target_dirname: str = "src") -> Path:
    """
    Search for the project root directory containing a specific subdirectory.
    """
    cur = Path(__file__).resolve()

    # Case 1: the current directory has a 'src'
    if (cur.parent / target_dirname).is_dir():
        return cur.parent

    # Case 2: search upwards if any parent has 'src'
    for parent in cur.parents:
        if (parent / target_dirname).is_dir():
            return parent

    raise RuntimeError(f"No parent directory found with '{target_dirname}' as subdirectory.")


BASE_PERSISTENCE_DIR = find_project_root() / "persistence"
BASE_PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)

class PersistenceManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, obj: Any, *, filename: Optional[str], namer: Callable[[Any], str]) -> Path:
        if filename is None:
            if namer is None:
                raise ValueError("filename or namer must be provided")
            filename = namer(obj)
        if not filename.endswith(".json"):
            filename += ".json"
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)  # <-- FIX: ensure parent dirs exist
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj.to_dict(), f, indent=2)
        return path


    def load(self, filename: str, from_dict: Callable[[dict], T]) -> T:
        if not filename.endswith(".json"):
            filename += ".json"
        path = self.base_dir / filename
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return from_dict(data)

    def list(self) -> list[str]:
        return [p.stem for p in self.base_dir.glob("*.json")]

# --------- Tiny per-class cache -------------
_PM_CACHE: dict[Type, PersistenceManager] = {}

def get_class_persistence_manager(cls: Type, *, subdir: Optional[str] = None) -> PersistenceManager:
    """
    Return (and cache) a PersistenceManager for a given class.
    subdir defaults to cls.__name__.lower().
    """
    key = (cls, subdir)
    if key not in _PM_CACHE:
        folder = BASE_PERSISTENCE_DIR / (subdir or cls.__name__.lower())
        _PM_CACHE[key] = PersistenceManager(folder)
    return _PM_CACHE[key]

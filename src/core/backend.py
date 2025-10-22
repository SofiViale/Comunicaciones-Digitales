"""
Backend abstraction layer
============================

Provides a **single object** that each high-level block (modulator,
demodulator, synchroniser, …) keeps as `self.backend`.  The object exposes:

* ``xp`` - NumPy-like array module (`numpy`, `cupy`, `jax.numpy`, …).
* ``fft(arr, axis=-1, plan=None)`` - thin wrapper around the backend's FFT
  (optionally using a cached *plan* on GPUs).
* ``plan_fft(n, batch=1)`` - return a backend-specific FFT plan/handle or
  ``None`` if not supported.
* ``stream()`` - current CUDA stream (``None`` on CPU back-ends).

This design eliminates all *"if xp is cupy"* checks from the DSP logic, keeps
GPU memory persistent, and allows future extensions (ROCm, oneAPI) by adding a
new subclass only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from types import ModuleType
from typing import Any, Optional, Union, Sequence

import importlib
import numpy as _np

try:
    _cp: Optional[ModuleType] = importlib.import_module("cupy")  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - CPU-only hosts
    _cp = None

# ------------------------------------------------------------
# Helper: ensure NVIDIA DLLs on Windows are visible **once**
# ------------------------------------------------------------
import os
def _patch_windows_cuda_dll_search_path():
    """Add CuPy-packaged CUDA DLL folders to the DLL search path (Windows only)."""
    if os.name != "nt":
        return
    import site
    from pathlib import Path

    for sub in ("cuda_nvrtc", "cufft"):
        for base in site.getsitepackages():
            bdir = Path(base) / "nvidia" / sub / "bin"
            if bdir.is_dir():
                os.add_dll_directory(str(bdir))


def _patch_linux_cuda_so_search():
    """Pre-carga manual de bibliotecas CUDA para CuPy en Linux (busca recursivamente en el venv)."""
    import os
    import ctypes
    from pathlib import Path

    if os.name != "posix":
        return

    os.environ.setdefault("CUDA_PATH", "/usr")

    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return

    root = Path(venv)

    patterns = ("libnvrtc.so*", "libcufft.so*", "libcurand.so*")

    for pattern in patterns:
        for so_path in root.rglob(pattern):
            try:
                ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # Ignora si falla la carga de alguno
                pass


_patch_windows_cuda_dll_search_path()  # ensure CUDA DLLs are visible on Windows
_patch_linux_cuda_so_search()  # ensure CUDA shared objects are visible on Linux
# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class AbstractBackend(ABC):
    """Common interface every back-end must implement."""

    #: Human-readable name ("numpy", "cupy", …)
    name: str

    #: NumPy-compatible module (numpy, cupy, jax.numpy, …)
    xp: Any

    # ................................................................. FFT
    def fft(self, arr, *, axis: int = -1, plan=None):  # noqa: D401
        """FFT wrapper.  *plan* is ignored by CPU back-ends."""
        return self.xp.fft.fft(arr, axis=axis)

    # ................................................................. plans
    def plan_fft(self, n: int, /, *, batch: int = 1):  # noqa: D401
        """Return a pre-computed FFT plan or *None* if unsupported."""
        return None

    # ................................................................. streams
    def stream(self):  # noqa: D401
        """Return current compute stream (CUDA) or *None* (CPU)."""
        return None

    # ................................................................. helpers
    def asnumpy(self, arr):  # noqa: D401
        """Return a *host* view of *arr* with zero copy when possible."""
        if self.name == "cupy":
            return self.xp.asnumpy(arr)
        return arr

    def clear_memory(self):  # noqa: D401
        """Clear all resources and memory used by this back-end."""
        """This is a no-op for CPU back-ends."""
        pass

    def as_strided(self, buf, shape: Sequence[int], strides: Sequence[int]):
        """Return a strided view of *buf* with the given *shape* and *strides*."""
        raise NotImplementedError

# ---------------------------------------------------------------------------
# NumPy (CPU) implementation
# ---------------------------------------------------------------------------

class NumpyBackend(AbstractBackend):
    """Pure-CPU back-end using NumPy."""

    name = "numpy"
    xp = _np

    def as_strided(self, arr, *, shape: Sequence[int], strides: Sequence[int]):
        from numpy.lib.stride_tricks import as_strided as _np_as_strided
        shape = tuple(int(s) for s in shape)
        strides = tuple(int(s) for s in strides)
        try:
            # NumPy >= 1.20 supports writeable flag
            return _np_as_strided(arr, shape=shape, strides=strides, writeable=False)
        except TypeError:
            # Older NumPy: no 'writeable' kw
            return _np_as_strided(arr, shape=shape, strides=strides)

# ----------------------------------------------------------------------
#  GPU backend (CuPy)
# ----------------------------------------------------------------------
class CupyBackend(AbstractBackend):
    """
    NumPy-like backend that targets a CUDA GPU via CuPy.
    """

    name = "cupy"
    xp   = __import__("cupy")
    _PLAN_CACHE: dict[tuple[int, int, int], "object"] = {}

    # ------------------------ construction ----------------------------

    def __init__(self, stream=None):
        import cupy as cp
        self._stream = stream or cp.cuda.Stream(null=True)
        self._fft_mod, self._plan_factory, self._plan_axes = self._locate_fft_api()

    # --------------------- locate plan helper -------------------------
    def _locate_fft_api(self):
        """
        Return (fft_module, plan_factory | None, plan_accepts_axes).

        * CuPy ≥13.3: (cupyx.scipy.fft , get_fft_plan , True )
        * CuPy 13.0-13.2: (cupy.fft      , old_get_plan , False)
        * CuPy ≤12:      (cupy.fft      , None         , False)
        """
        import cupy as cp
        try:                                 # >= 13.3
            import cupyx.scipy.fft as spfft
            if hasattr(spfft, "get_fft_plan"):
                return spfft, spfft.get_fft_plan, True
        except ImportError:
            pass
        old = getattr(getattr(cp.fft, "config", None), "get_plan", None)
        return cp.fft, old, False


    # ----------------------------- plan -------------------------------------
    def plan_fft(self, n: int, batch: int = 1):
        key = (n, batch, self._stream.ptr)
        if key in self._PLAN_CACHE:
            return self._PLAN_CACHE[key]

        if self._plan_factory is None:                           # CuPy ≤12
            from cupy.cuda import cufft
            plan = cufft.Plan1d(n, cufft.CUFFT_C2C, batch)
            if hasattr(plan, "set_stream"):
                plan.set_stream(self._stream)
        else:
            dummy = self.xp.zeros((batch, n), dtype=self.xp.complex64)
            if self._plan_axes:                                  # >=13.3
                plan = self._plan_factory(dummy, axes=(-1,), value_type='C2C')
            else:                                                # 13.0-13.2
                plan = self._plan_factory(dummy, plan_kw={"stream": self._stream})

        self._PLAN_CACHE[key] = plan
        return plan
    # ----------------------------- fft-------------------------------------
    def fft(self, arr, *, axis: int = -1):
        """
        Execute 1-D FFT along *axis* using this backend’s stream
        and an optional cuFFT *plan*.
        """
        with self._stream:
            return self._fft_mod.fft(arr, axis=axis)

    def clear_memory(self):
        xp = self.xp
        if hasattr(xp, "get_default_memory_pool"):
            """Clear all GPU memory used by this back-end."""
            pool = xp.get_default_memory_pool()
            if pool is not None:
                pool.free_all_blocks()

    def as_strided(self, arr, *, shape: Sequence[int], strides: Sequence[int]):
        from cupy.lib.stride_tricks import as_strided as _cp_as_strided
        shape = tuple(int(s) for s in shape)
        strides = tuple(int(s) for s in strides)
        with self._stream:
            try:
                return _cp_as_strided(arr, shape=shape, strides=strides, writeable=False)
            except TypeError:
                return _cp_as_strided(arr, shape=shape, strides=strides)

# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    if _cp is None:
        return False
    try:
        _cp.cuda.runtime.getDevice()  # type: ignore[attr-defined]
        return True
    except _cp.cuda.runtime.CUDARuntimeError:  # type: ignore[attr-defined]
        return False


def choose_backend(spec: Union[str, AbstractBackend, None] = "auto") -> AbstractBackend:
    """Return a concrete :class:`AbstractBackend`.

    * ``"auto"`` *(default)* - pick :class:`CupyBackend` if CUDA present,
      else :class:`NumpyBackend`.
    * ``"numpy"`` / ``"cupy"`` - explicit choice.
    * Pre-instantiated :class:`AbstractBackend` - returned as-is.
    """
    if isinstance(spec, AbstractBackend):  # already a backend object
        return spec

    if spec in (None, "auto"):
        return CupyBackend() if _cuda_available() else NumpyBackend()

    if spec == "numpy":
        return NumpyBackend()

    if spec == "cupy":
        if not _cuda_available():
            raise RuntimeError("CUDA device not available or CuPy missing")
        return CupyBackend()

    raise TypeError(f"Unsupported backend specifier: {spec!r}")

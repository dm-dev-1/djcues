"""Hardware-device selection for djcues's local-ML analysis
(--refine-drops --deep's Demucs pass in drop_enhance.py, beatgrid's
real-audio beat_this pass in beat_verify.py).

Zero-dependency at import time -- torch is imported lazily inside each
function, matching drop_enhance.py's own established discipline, so
importing this module never requires the `ml` extra.

Two-function contract, deliberately kept separate:
- probe_device()/list_available_backends() always describe reality --
  safe to call even when torch/an accelerator isn't installed, used by
  `djcues auth device` and the dashboard's /api/devices to show the
  user what's actually available, including the "not installed" case.
- resolve_device() gives you something *usable* right now: an explicit
  preference that fails degrades to cpu automatically rather than
  raising, so a caller never has to handle a device-unavailable error
  itself. Its one precondition -- torch must already be importable --
  is guaranteed in the real CLI/dashboard/batch-script flows because
  _check_refine_drops_available() (cli.py) already confirmed that
  before resolve_device() is ever called; this module doesn't
  duplicate that check.

resolve_device() is meant to be called exactly ONCE per run (CLI
invocation / dashboard job / batch-script process), before any
per-track work starts -- not once per track. Its result is memoized per
preference for the lifetime of the process specifically so a doomed
device is never re-probed on track 2, 3, ... 328 of a batch; see
resolve_device()'s own docstring for the full reasoning. A live driver
recovery mid-process is a deliberately accepted trade-off (speed only,
never correctness) -- force_recheck=True/reset_probe_cache() exist for
an explicit "recheck" action, not for automatic use.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

VALID_DEVICES = ("cpu", "cuda", "directml")
VALID_PREFERENCES = ("auto",) + VALID_DEVICES
# Order "auto" tries non-cpu backends in, before giving up and using cpu.
_AUTO_PROBE_ORDER = ("cuda", "directml")
_PROBE_TIMEOUT_S = 15.0

_logger = logging.getLogger("djcues.device")
# Standard library-code pattern (Python logging HOWTO): without this,
# a WARNING-level log with no application-configured handler falls
# through to logging.lastResort and prints to stderr on its own --
# confirmed live via `djcues auth device`, where that produced a raw,
# unformatted duplicate of the nicer click.echo warning right below it.
# The NullHandler silences that path while leaving the logger fully
# usable for anything that explicitly wants it (e.g. a future verbose
# mode) -- callers should keep surfacing user-facing messages via
# click.echo themselves, as _resolve_device_for_run (cli.py) already does.
_logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class DeviceProbeResult:
    device: str
    ok: bool
    error: str | None
    torch_device: Any = None


@dataclass(frozen=True)
class ResolvedDevice:
    requested: str
    active: str
    fell_back: bool
    reason: str | None
    torch_device: Any


def _run_with_timeout(fn, timeout: float) -> Any:
    """Run fn() on a daemon thread, bounded by timeout -- a daemon
    thread specifically: confirmed live in this project that
    concurrent.futures.ThreadPoolExecutor workers are NOT daemon
    threads by default in this Python, so a genuinely wedged GPU call
    could otherwise keep the process alive past this function's own
    return. Mirrors server.py's _DbWorker.run() bounded-wait shape
    (raw Thread + threading.Event), not ThreadPoolExecutor.

    Raises TimeoutError if fn doesn't complete in time, or re-raises
    whatever fn itself raised.
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def _target() -> None:
        try:
            box["value"] = fn()
        except Exception as e:  # noqa: BLE001 -- re-raised on the caller's thread below
            box["error"] = e
        finally:
            done.set()

    threading.Thread(target=_target, daemon=True).start()
    if not done.wait(timeout):
        raise TimeoutError(f"operation timed out after {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _probe_cpu() -> DeviceProbeResult:
    try:
        import torch
    except ImportError:
        return DeviceProbeResult("cpu", ok=False, error="torch is not installed (pip install djcues[ml])")

    try:
        def _op() -> Any:
            return torch.zeros(8) + 1

        _run_with_timeout(_op, _PROBE_TIMEOUT_S)
        return DeviceProbeResult("cpu", ok=True, error=None, torch_device=torch.device("cpu"))
    except Exception as e:  # noqa: BLE001 -- surfaced in DeviceProbeResult.error, not raised
        return DeviceProbeResult("cpu", ok=False, error=f"{type(e).__name__}: {e}")


def _probe_cuda() -> DeviceProbeResult:
    try:
        import torch
    except ImportError:
        return DeviceProbeResult("cuda", ok=False, error="torch is not installed (pip install djcues[ml])")

    if not torch.cuda.is_available():
        # Differentiated on purpose -- these are genuinely different fixes.
        # Confirmed live: requesting a CUDA tensor on a CPU-only torch build
        # raises AssertionError, not RuntimeError, so is_available() is the
        # only safe way to distinguish "no CUDA support compiled in" from
        # "CUDA build present, GPU unusable" before even attempting an op.
        if torch.version.cuda is None:
            error = (
                "this torch build has no CUDA support compiled in -- reinstall "
                "torch from a CUDA-enabled index (see "
                "https://pytorch.org/get-started/locally/) to use --device cuda"
            )
        else:
            error = (
                f"torch supports CUDA {torch.version.cuda} but no working GPU "
                "was found (driver, hardware, or another process holding it)"
            )
        return DeviceProbeResult("cuda", ok=False, error=error)

    try:
        def _op() -> Any:
            t = torch.zeros(8, device="cuda")
            return (t + 1).cpu()

        _run_with_timeout(_op, _PROBE_TIMEOUT_S)
        return DeviceProbeResult("cuda", ok=True, error=None, torch_device=torch.device("cuda"))
    except TimeoutError:
        return DeviceProbeResult(
            "cuda", ok=False,
            error=f"device probe timed out after {_PROBE_TIMEOUT_S}s -- possible driver hang",
        )
    except Exception as e:  # noqa: BLE001 -- surfaced in DeviceProbeResult.error, not raised
        return DeviceProbeResult("cuda", ok=False, error=f"{type(e).__name__}: {e}")


def _probe_directml() -> DeviceProbeResult:
    try:
        import torch_directml  # noqa: F401
    except ImportError:
        # Confirmed live against PyPI's real package index (2026-09-11):
        # torch-directml's latest release (0.2.5.dev240914, itself a
        # perpetual "dev" build with no newer release since) ships wheels
        # for cp38-cp312 only -- there is no build for Python 3.13+ at
        # all, on any platform. This isn't "not installed yet", it's
        # "cannot currently be installed" for anyone on Python 3.13+,
        # which `pip install djcues[directml]` alone would only reveal
        # after a confusing "no matching distribution" failure -- so
        # check for that specific, common case and say so directly
        # instead of pointing at a command that's doomed to fail the
        # same way. Deliberately phrased as "has not historically"
        # rather than a permanent claim, in case a future torch-directml
        # release adds 3.13+ support.
        import sys

        if sys.version_info >= (3, 13):
            return DeviceProbeResult(
                "directml", ok=False,
                error=(
                    f"torch_directml is not installed, and its latest release has not "
                    f"historically shipped a build for Python {sys.version_info.major}."
                    f"{sys.version_info.minor} -- check https://pypi.org/project/torch-directml/ "
                    f"for current support before trying `pip install djcues[directml]`"
                ),
            )
        return DeviceProbeResult(
            "directml", ok=False,
            error="torch_directml is not installed (pip install djcues[directml])",
        )
    if not torch_directml.is_available():
        return DeviceProbeResult(
            "directml", ok=False,
            error="torch_directml is installed but reports no available DirectML device",
        )

    try:
        import torch

        def _op() -> Any:
            dml_device = torch_directml.device()
            t = torch.zeros(8, device=dml_device)
            return (t + 1).cpu()

        _run_with_timeout(_op, _PROBE_TIMEOUT_S)
    except TimeoutError:
        return DeviceProbeResult(
            "directml", ok=False,
            error=f"device probe timed out after {_PROBE_TIMEOUT_S}s -- possible driver hang",
        )
    except Exception as e:  # noqa: BLE001 -- surfaced in DeviceProbeResult.error, not raised
        return DeviceProbeResult("directml", ok=False, error=f"{type(e).__name__}: {e}")

    # The device itself is real and works -- confirmed live (separate
    # verification venv, Python 3.12) on Intel Iris Xe Graphics: ordinary
    # tensor ops and a 2000x2000 matmul both compute correctly. But
    # djcues's two ML models both crash unrecoverably on this backend:
    # htdemucs (drop_enhance.py) and beat_this's LogMelSpect frontend
    # (beat_verify.py) each hit their STFT step and abort with "Invalid
    # or unsupported data type ComplexFloat" -- DirectML has no
    # complex-tensor support, and both models depend on it. Confirmed
    # live that this is NOT a catchable Python exception: it's a
    # native-level fatal abort that kills the whole process before any
    # try/except can run (a print() placed immediately before the
    # failing call never reached its own except clause afterward). That
    # rules out the retry-on-cpu fallback used for cuda -- there's
    # nothing left to retry once the process is already dead. So this
    # probe deliberately always reports ok=False: accurate about the
    # hardware being real and detected, while permanently keeping
    # resolve_device() from ever selecting it as the active device for
    # real work. This is a structural gap in DirectML's current op
    # coverage, not a transient condition worth re-checking later.
    try:
        # Confirmed live: torch_directml.device_name(0) can come back
        # with a trailing NUL byte baked in ("Intel(R) Iris(R) Xe
        # Graphics\x00") -- a raw C-string buffer, not python-side
        # whitespace, so plain .strip() alone doesn't remove it.
        hardware_name = torch_directml.device_name(0).replace("\x00", "").strip()
        detected = f" ({hardware_name} detected)"
    except Exception:  # noqa: BLE001 -- hardware name is cosmetic only
        detected = ""

    return DeviceProbeResult(
        "directml", ok=False,
        error=(
            f"torch_directml is installed and its device responds correctly to "
            f"ordinary tensor operations{detected}, but djcues's analysis models "
            f"(Demucs, beat_this) require STFT/complex-number operations that this "
            f"DirectML build cannot run -- confirmed live, both crash the whole "
            f"process outright rather than raising a normal error, so this device "
            f"can never be used for djcues's actual analysis and always falls back "
            f"to cpu"
        ),
    )


def probe_device(name: str) -> DeviceProbeResult:
    """Real smoke-test op on `name`, not just an is_available() check
    (which can lie -- e.g. a driver installed but broken). Always safe
    to call, including when torch/an accelerator isn't installed at
    all -- describes reality via DeviceProbeResult.ok/.error rather
    than raising.
    """
    if name == "cpu":
        return _probe_cpu()
    if name == "cuda":
        return _probe_cuda()
    if name == "directml":
        return _probe_directml()
    return DeviceProbeResult(name, ok=False, error=f"unknown device {name!r}, expected one of {VALID_DEVICES}")


def list_available_backends() -> list[DeviceProbeResult]:
    """probe_device() for every device in VALID_DEVICES, in order."""
    return [probe_device(d) for d in VALID_DEVICES]


_resolved_cache: dict[str, ResolvedDevice] = {}


def reset_probe_cache() -> None:
    """Clear resolve_device()'s memoization -- used by tests and by an
    explicit "recheck" action (e.g. a dashboard button). Does not touch
    the separate, once-per-process model caches in drop_enhance.py's
    _get_stem_separator()/beat_verify.py's _get_beat_tracker()."""
    _resolved_cache.clear()


def _log_resolution(resolved: ResolvedDevice) -> None:
    if resolved.fell_back:
        _logger.warning(
            "device %r unavailable (%s) -- falling back to cpu",
            resolved.requested, resolved.reason,
        )
    else:
        _logger.info("device resolved: %r -> %r", resolved.requested, resolved.active)


def _resolve_device_uncached(preference: str) -> ResolvedDevice:
    import torch

    if preference == "cpu":
        resolved = ResolvedDevice("cpu", "cpu", fell_back=False, reason=None, torch_device=torch.device("cpu"))
        _log_resolution(resolved)
        return resolved

    if preference == "auto":
        for candidate in _AUTO_PROBE_ORDER:
            probe = probe_device(candidate)
            if probe.ok:
                resolved = ResolvedDevice(
                    "auto", candidate, fell_back=False, reason=None, torch_device=probe.torch_device,
                )
                _log_resolution(resolved)
                return resolved
        # Not a "fallback" in the warn-worthy sense -- this is auto's
        # normal, expected resolution on a machine with no accelerator,
        # not an explicit request that failed. fell_back stays False so
        # callers don't warn about the common case.
        resolved = ResolvedDevice(
            "auto", "cpu", fell_back=False, reason="no accelerator available", torch_device=torch.device("cpu"),
        )
        _log_resolution(resolved)
        return resolved

    # Explicit "cuda"/"directml": this IS the warn-worthy case on failure.
    probe = probe_device(preference)
    if probe.ok:
        resolved = ResolvedDevice(preference, preference, fell_back=False, reason=None, torch_device=probe.torch_device)
    else:
        resolved = ResolvedDevice(preference, "cpu", fell_back=True, reason=probe.error, torch_device=torch.device("cpu"))
    _log_resolution(resolved)
    return resolved


def resolve_device(preference: str = "auto", *, force_recheck: bool = False) -> ResolvedDevice:
    """--device arg / configured preference -> the concrete device to
    actually use, with automatic CPU fallback on any failure. Call
    exactly once per run (CLI invocation / dashboard job / batch-script
    process), before any per-track work starts -- resolution is memoized
    per preference for the process's lifetime specifically so a doomed
    device is never re-probed on every track of a long batch, and so
    the one fallback warning (when relevant) is seen once, up front,
    rather than buried in per-track output.

    Precondition: torch must already be importable (the real CLI/
    dashboard/batch-script flows guarantee this via
    _check_refine_drops_available() running first) -- this function
    does not itself handle "torch isn't installed" gracefully, unlike
    probe_device(). force_recheck=True bypasses the cache.
    """
    if preference not in VALID_PREFERENCES:
        raise ValueError(f"unknown device preference {preference!r}, expected one of {VALID_PREFERENCES}")

    if not force_recheck and preference in _resolved_cache:
        return _resolved_cache[preference]

    result = _resolve_device_uncached(preference)
    _resolved_cache[preference] = result
    return result

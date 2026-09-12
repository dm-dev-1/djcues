"""Tests for djcues.device -- hardware-device selection for --deep's
Demucs pass (drop_enhance.py) and beatgrid's beat_this pass
(beat_verify.py).

Real torch throughout (no mocking torch itself away) -- this venv has a
genuine CPU-only 2.14.0+cpu build with no CUDA and no torch_directml
installed, which is exactly the environment several of these tests are
written to exercise for real rather than simulate. Where CUDA
"available but broken" needs testing, torch.cuda.is_available is
monkeypatched to True and the real op is left to fail on its own --
confirmed live (see the project plan) that this raises AssertionError
on this build, not a mocked stand-in for that behavior.
"""

from __future__ import annotations

import collections
import sys
import time
from unittest.mock import patch

import pytest

from djcues import device


@pytest.fixture(autouse=True)
def _reset_cache():
    """resolve_device()'s memoization is itself under test here, unlike
    drop_enhance.py's/beat_verify.py's incidental, never-directly-tested
    model caches -- must start clean every test."""
    device.reset_probe_cache()
    yield
    device.reset_probe_cache()


# --- probe_device -----------------------------------------------------


def test_probe_cpu_is_always_ok():
    result = device.probe_device("cpu")
    assert result.ok is True
    assert result.error is None
    assert result.torch_device is not None


def test_probe_cuda_unavailable_on_this_real_cpu_only_build():
    result = device.probe_device("cuda")
    assert result.ok is False
    assert "CUDA" in result.error


def test_probe_cuda_message_distinguishes_no_cuda_build_from_no_gpu(monkeypatch):
    import torch

    # torch.version.cuda is None on this real build -- the "reinstall
    # from a CUDA index" message, not the "GPU not found" one.
    assert torch.version.cuda is None
    result = device.probe_device("cuda")
    assert "no CUDA support compiled in" in result.error

    # Simulate a CUDA-enabled build (torch.version.cuda set) that still
    # has no working GPU -- the other message.
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result2 = device.probe_device("cuda")
    assert "no working GPU was found" in result2.error


def test_probe_cuda_is_available_lying_still_fails_the_real_op(monkeypatch):
    """The concrete regression case: is_available() alone can lie (e.g.
    a driver installed but broken) -- probe_device() must not trust it
    without a real op. Confirmed live on this machine: forcing
    is_available() True and then actually requesting a CUDA tensor
    raises AssertionError (torch not compiled with CUDA), not
    RuntimeError -- this test exercises that real failure, not a mock
    standing in for it."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = device.probe_device("cuda")
    assert result.ok is False
    assert result.error  # some real exception message, not silently swallowed


_VersionInfo = collections.namedtuple("version_info", ["major", "minor", "micro", "releaselevel", "serial"])


def test_probe_directml_not_installed_pre_313(monkeypatch):
    """The generic message -- Python version doesn't explain the
    absence, so just point at the install command."""
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    monkeypatch.setattr(sys, "version_info", _VersionInfo(3, 12, 0, "final", 0))
    result = device.probe_device("directml")
    assert result.ok is False
    assert "torch_directml is not installed" in result.error
    assert "djcues[directml]" in result.error
    assert "3.13" not in result.error


def test_probe_directml_not_installed_on_313_names_the_real_reason(monkeypatch):
    """Confirmed live against PyPI (see device.py's own comment):
    torch-directml's latest release ships no build at all for Python
    3.13+ -- pointing a 3.13+ user at `pip install djcues[directml]`
    without saying so would just send them into the same "no matching
    distribution" failure with no explanation. This is the actual
    situation on this project's own real dev machine, not a
    hypothetical -- so it's worth testing precisely, not just generically."""
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    monkeypatch.setattr(sys, "version_info", _VersionInfo(3, 13, 0, "final", 0))
    result = device.probe_device("directml")
    assert result.ok is False
    assert "3.13" in result.error
    assert "has not historically shipped a build" in result.error


def _fake_torch_directml(*, available=True, device_value="cpu", hardware_name="Fake GPU\x00"):
    """A stand-in torch_directml module. device_value="cpu" makes the
    probe's real op (torch.zeros(..., device=dml_device)) actually
    execute for real against this test env's real torch -- exercising
    device.py's own branching logic without needing torch_directml
    itself installed here. The real "DirectML device works but Demucs/
    beat_this crash on it" finding was verified live in a separate
    venv (see device.py's own comment on _probe_directml) and can't be
    reproduced by this CPU-only test environment. hardware_name defaults
    with a trailing NUL byte -- confirmed live that the real
    torch_directml.device_name() returns one ("Intel(R) Iris(R) Xe
    Graphics\x00"), which plain .strip() alone does not remove."""
    fake_module = type(sys)("torch_directml")
    fake_module.is_available = lambda: available
    fake_module.device = lambda: device_value
    fake_module.device_count = lambda: 1
    fake_module.device_name = lambda i=0: hardware_name
    return fake_module


def test_probe_directml_importable_but_reports_unavailable(monkeypatch):
    """torch_directml is installed but its own is_available() says no
    usable device -- the cheap pre-check short-circuits before
    attempting any real op, mirroring cuda's probe shape."""
    monkeypatch.setitem(sys.modules, "torch_directml", _fake_torch_directml(available=False))
    result = device.probe_device("directml")
    assert result.ok is False
    assert "no available DirectML device" in result.error


def test_probe_directml_is_available_lying_still_fails_the_real_op(monkeypatch):
    """Same "don't trust is_available() alone" guarantee as cuda's
    probe: is_available() reports True but the device value it hands
    back doesn't actually work, so the real op raises for real."""
    monkeypatch.setitem(
        sys.modules, "torch_directml",
        _fake_torch_directml(available=True, device_value="not-a-real-device"),
    )
    result = device.probe_device("directml")
    assert result.ok is False
    assert result.error


def test_probe_directml_working_device_still_reports_not_ok(monkeypatch):
    """The deliberate, permanent invariant of this probe: even when the
    DirectML device itself is real and responds correctly to ordinary
    tensor ops, probe_device("directml") must still report ok=False.
    Confirmed live (separate verification venv, real Intel Iris Xe
    Graphics hardware): djcues's actual models (Demucs, beat_this) both
    crash the whole process outright on this backend -- not a catchable
    Python exception, a native abort -- because neither DirectML's
    complex-tensor support exists nor can the usual retry-on-cpu
    fallback run once the process is already dead. So a generically
    "working" DirectML device must never become resolve_device()'s
    active device for real analysis work."""
    monkeypatch.setitem(
        sys.modules, "torch_directml",
        _fake_torch_directml(available=True, device_value="cpu", hardware_name="Fake Iris Xe\x00"),
    )
    result = device.probe_device("directml")
    assert result.ok is False
    assert result.torch_device is None
    assert "Fake Iris Xe detected" in result.error  # NUL stripped, no stray double space
    assert "\x00" not in result.error
    assert "complex" in result.error.lower()


def test_probe_directml_never_selected_by_auto_even_when_device_works(monkeypatch):
    """End-to-end version of the invariant above, through resolve_device
    -- a real digital fake of "DirectML works generically" must still
    not become auto's active device, matching the documented, permanent
    exclusion (this is not the same as
    test_resolve_auto_falls_through_cuda_to_directml, which mocks
    probe_device entirely to test ordering in isolation; this one goes
    through the real _probe_directml body with a fake torch_directml
    module standing in underneath it)."""
    monkeypatch.setitem(
        sys.modules, "torch_directml",
        _fake_torch_directml(available=True, device_value="cpu"),
    )

    def _cuda_unavailable(name):
        if name == "cuda":
            return device.DeviceProbeResult("cuda", ok=False, error="no cuda here")
        return device._probe_directml() if name == "directml" else device.probe_device(name)

    with patch.object(device, "probe_device", side_effect=_cuda_unavailable):
        resolved = device.resolve_device("auto")
    assert resolved.active == "cpu"


def test_probe_unknown_device_name():
    result = device.probe_device("quantum")
    assert result.ok is False
    assert "quantum" in result.error


def test_probe_cuda_timeout(monkeypatch):
    monkeypatch.setattr(device, "_PROBE_TIMEOUT_S", 0.05)

    def _slow_op():
        time.sleep(1.0)

    # Patch _run_with_timeout's callee indirectly: make torch.cuda
    # look available, then make the actual tensor op slow by swapping
    # in a sleeping stand-in for torch.zeros on the cuda path only is
    # awkward to isolate cleanly, so instead exercise _run_with_timeout
    # directly -- it's the exact mechanism probe_device relies on, and
    # this proves its timeout behavior in isolation.
    with pytest.raises(TimeoutError, match="timed out after 0.05s"):
        device._run_with_timeout(_slow_op, 0.05)


# --- list_available_backends -------------------------------------------


def test_list_available_backends_covers_every_valid_device():
    results = device.list_available_backends()
    assert [r.device for r in results] == list(device.VALID_DEVICES)


# --- resolve_device -----------------------------------------------------


def test_resolve_cpu_is_trivial_and_never_falls_back():
    resolved = device.resolve_device("cpu")
    assert resolved.requested == "cpu"
    assert resolved.active == "cpu"
    assert resolved.fell_back is False
    assert resolved.reason is None
    assert resolved.torch_device is not None


def test_resolve_auto_falls_through_to_cpu_with_no_accelerator_and_no_warning_flag():
    """auto finding nothing is the expected, common case on a machine
    with no GPU -- not a "fallback" in the warn-worthy sense (an
    explicit request that failed), so fell_back must stay False."""
    resolved = device.resolve_device("auto")
    assert resolved.active == "cpu"
    assert resolved.fell_back is False
    assert resolved.reason == "no accelerator available"


def test_resolve_explicit_unavailable_device_falls_back_with_reason():
    resolved = device.resolve_device("cuda")
    assert resolved.requested == "cuda"
    assert resolved.active == "cpu"
    assert resolved.fell_back is True
    assert resolved.reason  # the real probe error message


def test_resolve_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(
        device, "probe_device",
        lambda name: device.DeviceProbeResult(name, ok=(name == "cuda"), error=None if name == "cuda" else "no"),
    )
    resolved = device.resolve_device("auto")
    assert resolved.active == "cuda"
    assert resolved.fell_back is False


def test_resolve_auto_falls_through_cuda_to_directml(monkeypatch):
    monkeypatch.setattr(
        device, "probe_device",
        lambda name: device.DeviceProbeResult(name, ok=(name == "directml"), error=None if name == "directml" else "no"),
    )
    resolved = device.resolve_device("auto")
    assert resolved.active == "directml"


def test_resolve_invalid_preference_raises():
    with pytest.raises(ValueError, match="unknown device preference"):
        device.resolve_device("quantum")


def test_resolve_is_cached_per_preference_does_not_reprobe():
    """The literal "don't re-attempt a known-broken device on every
    track of a batch" guarantee -- a second call with the same
    preference must not invoke probe_device again."""
    calls = []
    real_probe = device.probe_device

    def _counting_probe(name):
        calls.append(name)
        return real_probe(name)

    with patch.object(device, "probe_device", side_effect=_counting_probe):
        device.resolve_device("cuda")
        assert calls == ["cuda"]
        device.resolve_device("cuda")
        assert calls == ["cuda"]  # unchanged -- second call hit the cache


def test_resolve_different_preferences_cached_independently():
    r1 = device.resolve_device("cuda")
    r2 = device.resolve_device("directml")
    assert r1.requested == "cuda"
    assert r2.requested == "directml"


def test_resolve_force_recheck_bypasses_cache():
    calls = []
    real_probe = device.probe_device

    def _counting_probe(name):
        calls.append(name)
        return real_probe(name)

    with patch.object(device, "probe_device", side_effect=_counting_probe):
        device.resolve_device("cuda")
        device.resolve_device("cuda", force_recheck=True)
        assert calls == ["cuda", "cuda"]


def test_reset_probe_cache_clears_memoization():
    calls = []
    real_probe = device.probe_device

    def _counting_probe(name):
        calls.append(name)
        return real_probe(name)

    with patch.object(device, "probe_device", side_effect=_counting_probe):
        device.resolve_device("cuda")
        device.reset_probe_cache()
        device.resolve_device("cuda")
        assert calls == ["cuda", "cuda"]

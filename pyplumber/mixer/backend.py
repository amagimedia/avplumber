"""Construction contract for one mixer backend shared by PGM, aux and wipes."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class MixerBackend(Protocol):
    name: str
    hardware_format: str

    def compositor(self, params, *, api=None): ...
    def transition(self, params): ...
    def scale(self, *, width=None, height=None, pixel_format=None, interpolation=None): ...
    def conversion(self, target, pixel_format, **options): ...
    def wipe_upload(self, color): ...


def mixer_backend(backend=None) -> MixerBackend:
    if backend is None or backend == "cuda":
        from .backends.cuda import CudaMixerBackend
        return CudaMixerBackend()
    if isinstance(backend, str):
        raise ValueError(f"Unsupported mixer backend: {backend}")
    if not isinstance(backend, MixerBackend):
        raise TypeError("Expected a mixer backend name or implementation")
    return backend

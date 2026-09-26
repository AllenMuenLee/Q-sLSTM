# VQC selection shared by the QLSTM baseline and the Q-sLSTM cells.
# Each variant lives in its own module; experiments record the chosen name as `vqc` in their config.
# A config without a `vqc` entry predates the option and used "original".

from .vqc_bounded import VQC as BoundedVQC
from .vqc_original import VQC as OriginalVQC

DEFAULT_VQC = "original"
VQC_VARIANTS = {"original": OriginalVQC, "bounded": BoundedVQC}

VQC = OriginalVQC  # backward-compatible name for the original circuit


def get_vqc_class(name=DEFAULT_VQC):
    if name not in VQC_VARIANTS:
        raise ValueError(f"vqc must be one of {tuple(VQC_VARIANTS)}, got {name!r}")
    return VQC_VARIANTS[name]


def config_vqc(config):
    """VQC variant of a run config (or solar study dict); missing means the original circuit."""
    return config.get("vqc", DEFAULT_VQC)

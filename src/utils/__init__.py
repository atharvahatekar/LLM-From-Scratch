"""Utilities: checkpointing, plotting and pretrained GPT-2 weight loading."""

from .checkpoint import load_checkpoint, save_checkpoint
from .gpt2_weights import assign, load_weights_into_gpt
from .plotting import plot_losses

# NOTE: gpt_download.download_and_load_gpt2 is intentionally NOT imported here
# because it lazily requires tensorflow. Import it directly when needed:
#     from src.utils.gpt_download import download_and_load_gpt2

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "assign",
    "load_weights_into_gpt",
    "plot_losses",
]

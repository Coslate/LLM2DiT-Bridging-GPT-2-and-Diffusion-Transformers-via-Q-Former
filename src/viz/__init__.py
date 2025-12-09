"""
Visualization helpers: Q-Former attention capture and heatmaps.
"""

from .attention_visualization import (
    enable_qformer_cross_attention_patch,
    view_qformer_attn,
    setup_attention_capture,
    perform_attention_visualization,
)

__all__ = [
    "enable_qformer_cross_attention_patch",
    "view_qformer_attn",
    "setup_attention_capture",
    "perform_attention_visualization",
]

"""实验执行包：消融流水线 + 指标计算。"""

from .experiment import ExperimentRunner, run_ablation
from .metrics import cluster_boundaries, compute_metrics, format_metrics_table

__all__ = [
    "ExperimentRunner",
    "cluster_boundaries",
    "compute_metrics",
    "format_metrics_table",
    "run_ablation",
]

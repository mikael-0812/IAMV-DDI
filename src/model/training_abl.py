"""Multiclass ablation entry point.

The training loop is shared with ``training_multiclass`` so all ablations use
the same data split, optimization, evaluation, and checkpoint protocol.
"""

from src.model.training_multiclass import main


if __name__ == "__main__":
    main(
        default_ablation_mode="wo_2d",
        default_metric_best="acc",
    )

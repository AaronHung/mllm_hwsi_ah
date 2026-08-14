"""Paper-facing labels for the frozen internal method keys.

Internal keys remain stable so Protocol-v1 CSVs stay readable and immutable.
These labels are used by new reports, tables, and paper-facing documentation.
"""

from __future__ import annotations

METHOD_LABELS = {
    "seqft": "sequential fine-tuning",
    "ewc": "EWC parameter regularization",
    "lwf": "new-state old-policy distillation",
    "replay": "counterfactual-teacher replay",
    "distill": "old-policy / policy-fidelity distillation",
    "ours_uniform": "replay + policy distillation (uniform loss weight)",
    "ours": "Utility-Weighted Replay Distillation",
    "joint": "joint-training reference",
}


def method_label(key: str) -> str:
    """Return a paper-facing label without changing the CSV method key."""
    return METHOD_LABELS.get(key, key)

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field


@dataclass
class OptimizationParams:
  consider_all_permutations: bool = field(default=False)
  retrieve_all_labels: bool = field(default=False)
  disable_early_stopping: bool = field(default=False)
  disable_early_majority_detection: bool = field(default=False)
  disable_path_length_bounds: bool = field(default=False)
  disable_majority_pruning: bool = field(default=False)

  @classmethod
  def from_args(cls: type["OptimizationParams"], args: Namespace):
    """
    Initialize OptimizationParams from a Namespace object.
    """
    return cls(
      consider_all_permutations=getattr(args, "consider_all_permutations", False),
      retrieve_all_labels=getattr(args, "retrieve_all_labels", False),
      disable_early_stopping=getattr(args, "no_early_stopping", False),
      disable_early_majority_detection=getattr(args, "no_early_majority_detection", False),
      disable_path_length_bounds=getattr(args, "no_path_length_bounds", False),
      disable_majority_pruning=getattr(args, "no_majority_pruning", False),
    )

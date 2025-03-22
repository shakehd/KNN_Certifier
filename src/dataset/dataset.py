
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple, NotRequired, Required, Self, TypedDict
from numpy import vstack, hstack

from src.utils.base_types import ArrayNxM, NDVector


class DatasetParams(TypedDict, total=False):
  format: Required[str]
  training_set: Required[str]
  test_set: Required[str]
  category_indexes: NotRequired[list[int]]
  perturb_categories: NotRequired[list[int]]

@dataclass(frozen=True)
class CatFeature:
  idx: int
  size: int
  perturb: bool

# @dataclass
# class DatasetProps:
#   columns: ClassVar[int] = 0
#   cat_features: ClassVar[dict[int, CatFeature]] = {}
#   num_features_ix: ClassVar[list[int]] = []

#   num_features_start_ix: ClassVar[int] = field(default=0)

#   @classmethod
#   def has_categories(cls: type[DatasetProps]) -> bool:
#     return bool(cls.cat_features)

# @dataclass
# class Dataset:
#   points: ArrayNxM
#   labels: NDVector

@dataclass
class DatasetProps:
  columns: int = field(default_factory=int)
  cat_features: dict[int, CatFeature] = field(default_factory=dict)
  num_features_ix: list[int] = field(default_factory=list)

  num_features_start_ix: int = field(default=0)


  def has_categories(self: Self) -> bool:
    return bool(self.cat_features)

@dataclass
class Dataset:
  points: ArrayNxM
  labels: NDVector

  num_points: int = field(init=False)


  def __post_init__(self: Self) -> None:
    self.num_points = self.points.shape[0]

  def __getitem__(self: Self, key: Any) -> Dataset:
    return Dataset(self.points[key], self.labels[key])

  @classmethod
  def vstack(cls: type[Dataset], *datasets: Dataset) -> Dataset:
    points: list[ArrayNxM] = [d.points for d in datasets]
    labels: list[ArrayNxM] = [d.labels for d in datasets]

    return cls(vstack(points), hstack(labels)) # type: ignore
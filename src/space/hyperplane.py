from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List,  Self

from sklearn.metrics import DistanceMetric
from src.utils.base_types import Array1xN, NDVector
import numpy as np

@dataclass(init=False)
class Hyperplane:

    def __init__(self: Self, coefficients: NDVector | None = None,
                 constant: float = float(), dimensions: int =0):
        if coefficients is None:
            coefficients = np.zeros(dimensions)

        self.coefficients: NDVector = coefficients
        self.constant: float = constant

    def  distance_to_point(self: Self, point:NDVector,
                           distance_metric: DistanceMetric)-> float:

        norm_factor = abs((self.constant - np.dot(self.coefficients, point)))

        if 'EuclideanDistance' in type(distance_metric).__name__:
            return norm_factor / np.linalg.norm(self.coefficients)

        if 'Manhattan' in type(distance_metric).__name__:
             return norm_factor / np.max(np.abs(self.coefficients))
        else:
            raise ValueError(f'distance metric {type(distance_metric).__name__} not supported')

    def __call__(self:Self , point: NDVector, **kwds: Any) -> float:
        return np.dot(self.coefficients, point) - self.constant

    def __str__(self: Self) -> str:
        coefs_repr: List[str] = [f"{self.coefficients[i]} x_{i}"
                                 for i in range(0, self.coefficients.shape[0])]
        return ' + '.join(coefs_repr) + f" + {self.constant} = 0"

    @classmethod
    def build_equidistant_plane(cls: type[Hyperplane], p1: NDVector,
                                p2: NDVector) -> 'Hyperplane':
        coefs: Array1xN = 2 * (p2 - p1)
        const: float  = (p2**2).sum() - (p1**2).sum()

        return cls(coefs, const)



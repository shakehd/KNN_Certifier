from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Self
from numpy import sqrt
from sklearn.metrics import DistanceMetric
from pprint import pformat
from textwrap import indent
import numpy as np
import logging

from ..space.polyhedron import Polyhedron
from .dominance_graph import DominanceGraph
from ..dataset.dataset import Dataset
from ..perturbation.perturbation import Perturbation
from ..space.partition_tree import Partitions

logger = logging.getLogger(__name__)


@dataclass
class AbstractClassifier:
  partition_tree: Optional[Partitions] = field(default=None)

  point_number: ClassVar[int | None] = None

  def fit(self: Self,
          dataset: Dataset,
          partition_size: int = 20,
          random_state: Optional[int] = None) -> None:
    distance_metric: DistanceMetric = DistanceMetric.get_metric('euclidean') # type: ignore
    self.partition_tree = Partitions(dataset, distance_metric, partition_size, random_state)

  def get_classification(self: Self,
                         perturbation: Perturbation,
                         k_vals: list[int]) -> dict[int, set[int]]:

    max_k = max(k_vals)
    classifications: defaultdict[int, set[int]] = defaultdict(set)

    if self.partition_tree is None:
      raise ValueError("Missing dataset!! Call fit first with a dataset.")

    for adv_region in perturbation.get_adversarial_regions():

      logger.info("\t\tadversarial region: %s\n", adv_region)

      eq_lhs, eq_rhs = adv_region.get_equality_constraints()
      Polyhedron.bounds = adv_region.get_bounds()
      Polyhedron.equalities_lhs = eq_lhs
      Polyhedron.equalities_rhs = eq_rhs

      init_radius: float = 2*adv_region.epsilon*sqrt(adv_region.point.shape[0])

      closer_points, dists = self.partition_tree.query_point(adv_region.point, init_radius, max_k, True)

      closer_points = closer_points[dists <= dists[max_k-1] + init_radius]
      if closer_points.num_points < max_k:
        logger.error(f'Not enough closer points !!')

      logger.debug("\t closer points: ")
      logger.debug('%s\n', indent(pformat(closer_points.points, compact=True),'\t\t'))

      dominance_graph: DominanceGraph = DominanceGraph.build_dominance_graph(adv_region, closer_points)

      actual_points = np.array([v.point for v in dominance_graph.get_vertices()])
      actual_points = np.append(actual_points, [adv_region.lower_bound(), adv_region.upper_bound()], axis=0)

      possible_classifications = dominance_graph.get_neighbors_label( k_vals)

      logger.info('\t\tlabels: ')
      for k in range(max_k+1):
        if k in k_vals:
          logger.info(f'\t\t\tk = {k} -> {possible_classifications[k]}', )
          classifications[k] |= possible_classifications[k]
      logger.info('\n')

    return classifications
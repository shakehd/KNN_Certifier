from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain, combinations, permutations
from math import ceil
from pprint import pformat
from textwrap import indent
from typing import ClassVar, Optional, Self

import numpy as np
from numpy import sqrt
from scipy import optimize
from sklearn.metrics import DistanceMetric

from src.abstract.optimization_params import OptimizationParams
from src.utils.base_types import Array1xN, Label, NDVector
from src.utils.combinatotial import permutations_with_constraints

from ..dataset.dataset import Dataset
from ..perturbation.perturbation import Perturbation
from ..space.partition_tree import Partitions
from ..space.polyhedron import Polyhedron
from .dominance_graph import DominanceGraph, Vertex, VertexId

logger = logging.getLogger(__name__)


@dataclass
class Path:
  @classmethod
  def exist_valid_path(
    cls: type[Path],
    vertices: set[VertexId] | frozenset[VertexId],
    dom_graph: DominanceGraph,
  ) -> bool:
    all_predecessor: set[VertexId] = set(
      reduce(
        (lambda acc, val: acc.union(dom_graph[val].closer_vertices)),  # type: ignore
        vertices,
        set(),  # type: ignore
      )
    )  # type: ignore# type: ignore

    if not vertices >= all_predecessor:
      return False

    inequalities_lhs: list[Array1xN] = []
    inequalities_rhs: list[float] = []

    bisectors = dom_graph.bisectors
    for vertex_id in vertices:
      other_vertices_id = (_.id for _ in dom_graph.get_vertices() if _.id not in vertices)
      for other_vertex_id in other_vertices_id:
        if (vertex_id, other_vertex_id) in bisectors:
          bisector = bisectors[(vertex_id, other_vertex_id)]
          inequalities_lhs.append(bisector.coefficients)
          inequalities_rhs.append(bisector.constant)
        else:
          bisector = bisectors[(other_vertex_id, vertex_id)]
          inequalities_lhs.append(-bisector.coefficients)
          inequalities_rhs.append(-bisector.constant)

    return Polyhedron(np.array(inequalities_lhs), np.array(inequalities_rhs)).is_valid()  # type: ignore


@dataclass
class AbstractClassifier:
  opt_params: OptimizationParams
  partition_tree: Optional[Partitions] = field(default=None)

  point_number: ClassVar[int | None] = None

  def initialize(
    self: Self,
    dataset: Dataset,
    partition_size: int = 20,
    random_state: Optional[int] = None,
  ) -> None:
    distance_metric: DistanceMetric = DistanceMetric.get_metric("euclidean")  # type: ignore
    self.partition_tree = Partitions(dataset, distance_metric, partition_size, random_state)

  def get_classification(
    self: Self, perturbation: Perturbation, k_vals: list[int], all_labels: bool = True
  ) -> dict[int, set[int]]:
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

      init_radius: float = 2 * adv_region.epsilon * sqrt(adv_region.point.shape[0])

      closer_points, dists = self.partition_tree.query_point(adv_region.point, init_radius, max_k, True)

      closer_points = closer_points[dists <= dists[max_k - 1] + init_radius]
      if closer_points.num_points < max_k:
        logger.error("Not enough closer points !!")

      logger.debug("\t closer points: ")
      logger.debug("%s\n", indent(pformat(closer_points.points, compact=True), "\t\t"))

      dominance_graph: DominanceGraph = DominanceGraph.build_dominance_graph(adv_region, closer_points)

      # possible_classifications = dominance_graph.get_neighbors_label(k_vals)

      possible_classifications = self._get_neighbors_label(dominance_graph, k_vals)

      logger.info("\t\tlabels: ")
      for k in range(max_k + 1):
        if k in k_vals:
          logger.info(
            f"\t\t\tk = {k} -> {possible_classifications[k]}",
          )
          classifications[k] |= possible_classifications[k]
      logger.info("\n")

    return classifications

  def _get_neighbors_label(
    self: Self,
    dominance_graph: DominanceGraph,
    k_vals: list[int],
  ) -> dict[int, set[int]]:
    max_k: int = max(k_vals)
    classifications: dict[int, set[int]] = dict((k, set()) for k in k_vals)

    vertices_with_same_labels: dict[Label, list[Vertex]] = dominance_graph.get_vertices_with_same_labels()
    approx_upper_bounds: dict[Label, tuple[int, int]]

    if not self.opt_params.disable_path_length_bounds:
      approx_upper_bounds = dict(
        [
          (
            label,
            self._approx_max_path_length(dominance_graph, label, max_k, vertices_with_same_labels),
          )
          for label in vertices_with_same_labels
        ]
      )
    else:
      approx_upper_bounds = dict(
        [(label, (max_k, len(vertices_with_same_labels[label]))) for label in vertices_with_same_labels]
      )

    labels_to_exclude: set[Label] = set([label for label, val in approx_upper_bounds.items() if val[0] == 0])
    possible_labels_with_vertices: dict[Label, list[Vertex]] = vertices_with_same_labels
    labels = [l for l in vertices_with_same_labels if l not in labels_to_exclude]

    if len(possible_labels_with_vertices) == 1:
      for label_occur in k_vals:
        classifications[label_occur].add(dominance_graph["0"].label)

      return classifications

    vertices: list[Vertex] = [v for v in dominance_graph.get_vertices() if len(v.closer_vertices) < max_k]

    vertices = sorted(vertices, key=lambda val: len(val.closer_vertices))

    invalid_paths: set[tuple[VertexId, ...]] | None = None
    if self.opt_params.consider_all_permutations:
      invalid_paths = set()

    def get_possible_vertices(
      label: Label,
      label_counter: Counter[Label],
      existing_vertices: list[VertexId],
      other_vertices: list[Vertex],
      max_length: int,
    ) -> list[VertexId]:
      possible_vertices = [
        v.id
        for v in other_vertices
        if label_counter[v.label] < label_counter[label]
        and v.id not in existing_vertices
        and not ignore_vertex(
          label,
          label_counter[label],
          existing_vertices,
          v,
          max_length,
        )
      ]

      return possible_vertices

    def ignore_vertex(
      label: Label,
      label_count: int,
      existing_vertices: list[VertexId],
      vertex: Vertex,
      max_path_length: int,
    ) -> bool:
      all_vertices: set[VertexId] = set(existing_vertices) | vertex.closer_vertices

      if len(all_vertices) > max_path_length:
        return True

      lb_count = Counter(dominance_graph[v].label for v in all_vertices)
      lb_count.update([vertex.label])
      most_common = lb_count.most_common(1)[0]

      insufficient_vertex: bool = most_common[1] - lb_count[label] > label_count - lb_count[label]

      insufficient_length = most_common[1] - lb_count[label] > max_path_length - (len(vertex.closer_vertices) + 1)

      if insufficient_vertex or insufficient_length:
        return True

      return False

    def exists_path(label: Label, vertices: list[Vertex], max_length: int) -> tuple[bool, int]:
      all_vertices: set[VertexId] = set(
        reduce(
          (lambda acc, val: acc.union(val.closer_vertices | set([val.id]))),  # type: ignore
          vertices,
          set(),  # type: ignore
        )
      )  # type: ignore
      sorted_vertices: list[int] = list(sorted([int(v) for v in all_vertices]))
      vertex_ids: dict[int, int] = dict((v, idx) for idx, v in enumerate(sorted_vertices))

      label_vertices_ids = [vertex_ids[vid] for vid in sorted_vertices if dominance_graph[str(vid)].label == label]

      constraints: list[optimize.LinearConstraint] = []
      sizes: NDVector = np.full_like(sorted_vertices, 1)
      values: NDVector = np.full_like(sorted_vertices, 1)

      for idx, vertex_id in enumerate(sorted_vertices):
        vertex: Vertex = dominance_graph[str(vertex_id)]

        if vertex.label == label:
          values[idx] = 2

        if len(vertex.closer_vertices) > 0:
          for closer_id in vertex.closer_vertices:
            constraint: NDVector = np.full_like(sorted_vertices, 0)
            constraint[vertex_ids[int(closer_id)]] = 1
            constraint[idx] = -1
            constraints.append(optimize.LinearConstraint(A=constraint, lb=0, ub=1))

      for other_label in possible_labels_with_vertices:
        constraint: NDVector = np.full_like(sorted_vertices, 0)
        constraint[label_vertices_ids] = 1
        if other_label != label:
          constraint[
            [vertex_ids[int(v.id)] for v in possible_labels_with_vertices[other_label] if int(v.id) in vertex_ids]
          ] = -1
          constraints.append(optimize.LinearConstraint(A=constraint, lb=0, ub=len(label_vertices_ids)))

      bounds = optimize.Bounds(0, 1)
      integrality = np.full_like(sorted_vertices, True)
      capacity: int = max_length
      constraints.append(optimize.LinearConstraint(A=sizes, lb=capacity, ub=capacity))

      solution = optimize.milp(
        c=-values,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
      )

      if solution.success and solution.x is not None:
        vertex_solution = np.nonzero(solution.x)[0]
        label_vertex_added = len([_ for _ in vertex_solution if _ in label_vertices_ids])

        return label_vertex_added != 0, label_vertex_added

      return False, 0

    def handle_label_majority(
      label: Label,
      label_counter: Counter[Label],
      max_length: int,
      apply_early_stopping: bool,
      apply_early_majority: bool,
      apply_majority_prunning: bool,
    ) -> None:
      if max_length in k_vals:
        most_occurr = label_counter.most_common(1)[0][1]

        for maj_label in (l for l, v in label_counter.items() if v == most_occurr):
          classifications[max_length].add(maj_label)

      if apply_early_majority or apply_majority_prunning:
        for k in (_ for _ in range(max_length, max_k + 1) if _ in k_vals):
          classifications[k].add(label)

      if apply_early_stopping:
        raise StopIteration()

    for label in labels:
      if not self.opt_params.retrieve_all_labels and all([len(val) > 1 for val in classifications.values()]):
        break

      label_vertices = possible_labels_with_vertices[label]
      max_label_occur: int = approx_upper_bounds[label][1]

      other_vertices: list[Vertex] = list(
        chain(*[vertices for l, vertices in possible_labels_with_vertices.items() if l != label])
      )

      combinations_generator = permutations if self.opt_params.consider_all_permutations else combinations

      for max_length in [k for k in range(1, max_k + 1)]:
        if all([label in classifications[len] for len in range(max_length, max_k + 1) if len in k_vals]):
          break
        # if label in classifications[max_length]:
        #   continue

        possible_label_vertices = [
          v for v in label_vertices if not ignore_vertex(label, max_label_occur, [], v, max_length)
        ]

        exists, max_occur = exists_path(label, possible_label_vertices + other_vertices, max_length)

        if exists:
          majority_prunning_detected: bool = False
          early_majority_detected: bool = False

          try:
            init: int = min(max_length, max_occur)
            min_occur: int = ceil(max_length / len(vertices_with_same_labels.keys())) - 1
            labels_occurs: range = range(init, min_occur, -1)

            for label_occur in labels_occurs:
              possible_label_vertices = [
                v for v in label_vertices if not ignore_vertex(label, label_occur, [], v, max_length)
              ]

              for label_com in combinations(possible_label_vertices, label_occur):
                all_vertices: list[VertexId] = list(
                  reduce(
                    (lambda acc, val: acc.union(val.closer_vertices) | set([val.id])),  # type: ignore
                    label_com,  # type: ignore
                    set(),  # type: ignore
                  )  # type: ignore
                )

                label_count = Counter([dominance_graph[v].label for v in all_vertices])

                if (
                  len(all_vertices) <= max_length
                  and label_count[label] <= label_count.most_common(1)[0][1]
                  and label_count[label] == label_occur
                ):
                  if len(all_vertices) == max_length:
                    if self.opt_params.consider_all_permutations:
                      for vert_perturbation in permutations_with_constraints(
                        [(dominance_graph[vix].closer_vertices, vix) for vix in all_vertices],
                        max_length,
                        invalid_paths,  # type: ignore
                      ):
                        if self._exist_valid_path(
                          list(vert_perturbation), dominance_graph, self.opt_params.consider_all_permutations
                        ):
                          if label_occur == max_length:
                            early_majority_detected = max_length >= ceil(max_k / 2)
                            early_majority_detected = (
                              early_majority_detected and not self.opt_params.disable_early_majority_detection
                            )

                          handle_label_majority(
                            label,
                            label_count,
                            max_length,
                            not self.opt_params.disable_early_stopping,
                            early_majority_detected,
                            majority_prunning_detected and not self.opt_params.disable_majority_pruning,
                          )

                        else:
                          invalid_paths.add(tuple(vert_perturbation))  # type: ignore
                      else:
                        invalid_paths.add(tuple(all_vertices))  # type: ignore

                    else:
                      if self._exist_valid_path(
                        all_vertices, dominance_graph, self.opt_params.consider_all_permutations
                      ):
                        if label_occur == max_length:
                          early_majority_detected = max_length >= ceil(max_k / 2)
                          early_majority_detected = (
                            early_majority_detected and not self.opt_params.disable_early_majority_detection
                          )

                        if len(label_count) > 1:
                          most_commons = label_count.most_common(2)
                          majority_prunning_detected = most_commons[0][1] - most_commons[1][1] >= max_k - max_length
                          majority_prunning_detected = (
                            majority_prunning_detected and not self.opt_params.disable_majority_pruning
                          )

                        handle_label_majority(
                          label,
                          label_count,
                          max_length,
                          not self.opt_params.disable_early_stopping,
                          early_majority_detected,
                          majority_prunning_detected and not self.opt_params.disable_majority_pruning,
                        )

                  if len(all_vertices) < max_length:
                    other_possible_vertices_id = get_possible_vertices(
                      label,
                      label_count,
                      all_vertices,
                      other_vertices,
                      max_length,
                    )

                    if len(other_possible_vertices_id) > 0:
                      for other_label_com in combinations_generator(
                        other_possible_vertices_id,
                        max_length - len(all_vertices),
                      ):
                        if self.opt_params.consider_all_permutations:
                          for vert_perturbation in permutations_with_constraints(
                            [
                              (dominance_graph[vix].closer_vertices, vix)
                              for vix in all_vertices + list(other_label_com)
                            ],
                            max_length,
                            invalid_paths,  # type: ignore
                          ):
                            final_count = Counter(
                              [dominance_graph[v].label for v in vert_perturbation]  # type: ignore
                            )

                            if self._exist_valid_path(list(vert_perturbation), dominance_graph, True):
                              if final_count[label] >= final_count.most_common(1)[0][1]:
                                most_commons = final_count.most_common(2)
                                majority_prunning_detected = (
                                  most_commons[0][1] - most_commons[1][1] >= max_k - max_length
                                )
                                majority_prunning_detected = (
                                  majority_prunning_detected and not self.opt_params.disable_majority_pruning
                                )

                                handle_label_majority(
                                  label,
                                  final_count,
                                  max_length,
                                  not self.opt_params.disable_early_stopping,
                                  early_majority_detected and not self.opt_params.disable_early_majority_detection,
                                  majority_prunning_detected,
                                )

                            else:
                              invalid_paths.add(tuple(all_vertices))  # type: ignore

                        else:
                          final_count = label_count + Counter([dominance_graph[v].label for v in other_label_com])

                          if final_count[label] >= final_count.most_common(1)[0][1] and self._exist_valid_path(
                            all_vertices + [dominance_graph[v].id for v in other_label_com], dominance_graph, False
                          ):
                            most_commons = final_count.most_common(2)
                            majority_prunning_detected = most_commons[0][1] - most_commons[1][1] >= max_k - max_length
                            majority_prunning_detected = (
                              majority_prunning_detected and not self.opt_params.disable_majority_pruning
                            )

                            handle_label_majority(
                              label,
                              final_count,
                              max_length,
                              not self.opt_params.disable_early_stopping,
                              early_majority_detected and not self.opt_params.disable_early_majority_detection,
                              majority_prunning_detected,
                            )

          except StopIteration:
            if max_length == max_k:
              break

            # if (
            #   (early_majority_detected and not self.opt_params.disable_early_majority_detection)
            #   or (majority_prunning_detected and not self.opt_params.disable_majority_pruning)
            #   and (not self.opt_params.disable_early_stopping or max_length in k_vals)
            # ):

    return classifications

  def _approx_max_path_length(
    self: Self,
    dominance_graph: DominanceGraph,
    label: Label,
    max_length: int,
    vertices_with_same_labels: dict[Label, list[Vertex]],
  ) -> tuple[int, int]:
    vertices = list(vertices_with_same_labels.values())
    all_vertices: set[VertexId] = list(  # type: ignore
      reduce(
        (lambda acc, val: acc.union(val.closer_vertices | set([val.id]))),  # type: ignore
        chain(*vertices),
        set(),  # type: ignore
      )
    )  # type: ignore

    sorted_vertices: list[int] = np.array(list(sorted([int(v) for v in all_vertices])))  # type: ignore
    vertex_ids: dict[int, int] = dict((v, idx) for idx, v in enumerate(sorted_vertices))

    label_vertices_ids = [vertex_ids[vid] for vid in sorted_vertices if dominance_graph[str(vid)].label == label]

    constraints: list[optimize.LinearConstraint] = []
    sizes: NDVector = np.full_like(sorted_vertices, 1)
    values: NDVector = np.full_like(sorted_vertices, 1)

    other_label_constraint: NDVector = np.full_like(sorted_vertices, 0)
    for idx, vertex_id in enumerate(sorted_vertices):
      vertex: Vertex = dominance_graph[str(vertex_id)]

      if vertex.label == label:
        values[idx] = 2

      if len(vertex.closer_vertices) > 0:
        for closer_id in vertex.closer_vertices:
          constraint: NDVector = np.full_like(sorted_vertices, 0)
          constraint[vertex_ids[int(closer_id)]] = 1
          constraint[idx] = -1
          constraints.append(optimize.LinearConstraint(A=constraint, lb=0, ub=1))

    for alabel in vertices_with_same_labels:
      if alabel != label:
        other_label_constraint: NDVector = np.full_like(sorted_vertices, 0)
        other_label_constraint[label_vertices_ids] = 1
        other_label_constraint[
          [vertex_ids[int(v.id)] for v in vertices_with_same_labels[alabel] if int(v.id) in vertex_ids]
        ] = -1

        constraints.append(optimize.LinearConstraint(A=other_label_constraint, lb=0, ub=len(label_vertices_ids)))

    bounds = optimize.Bounds(0, 1)
    integrality = np.full_like(sorted_vertices, True)
    capacity: int = max_length
    constraints.append(optimize.LinearConstraint(A=sizes, lb=0, ub=capacity))

    solution: optimize.OptimizeResult = optimize.milp(
      c=-values, constraints=constraints, integrality=integrality, bounds=bounds
    )

    if solution.success and solution.x is not None:
      vertex_solution = np.nonzero(solution.x)[0]
      label_vertex_added = len([_ for _ in vertex_solution if _ in label_vertices_ids])
      if label_vertex_added != 0:
        return len(vertex_solution), label_vertex_added

    return 0, 0

  def _exist_valid_path(
    self: Self,
    vertices: list[VertexId],
    dom_graph: DominanceGraph,
    is_permitation: bool = False,
  ) -> bool:
    all_predecessor: set[VertexId] = set(
      reduce(
        (lambda acc, val: acc.union(dom_graph[val].closer_vertices)),  # type: ignore
        vertices,
        set(),  # type: ignore
      )
    )  # type: ignore# type: ignore

    if not set(vertices) >= all_predecessor:
      return False

    inequalities_lhs: list[Array1xN] = []
    inequalities_rhs: list[float] = []
    bisectors = dom_graph.bisectors

    if is_permitation:
      for i, vertex_i in enumerate(vertices):
        for j, vertex_j in enumerate(vertices):
          if j > i:
            if not set(vertices[:j]) >= dom_graph[vertex_i].closer_vertices:
              return False

            if (vertex_i, vertex_j) in bisectors:
              bisector = bisectors[(vertex_i, vertex_j)]
              inequalities_lhs.append(bisector.coefficients)
              inequalities_rhs.append(bisector.constant)
            else:
              bisector = bisectors[(vertex_j, vertex_i)]
              inequalities_lhs.append(-bisector.coefficients)
              inequalities_rhs.append(-bisector.constant)

    for vertex_id in vertices:
      other_vertices_id = (_.id for _ in dom_graph.get_vertices() if _.id not in vertices)
      for other_vertex_id in other_vertices_id:
        if (vertex_id, other_vertex_id) in bisectors:
          bisector = bisectors[(vertex_id, other_vertex_id)]
          inequalities_lhs.append(bisector.coefficients)
          inequalities_rhs.append(bisector.constant)
        else:
          bisector = bisectors[(other_vertex_id, vertex_id)]
          inequalities_lhs.append(-bisector.coefficients)
          inequalities_rhs.append(-bisector.constant)

    return Polyhedron(np.array(inequalities_lhs), np.array(inequalities_rhs)).is_valid()  # type: ignore

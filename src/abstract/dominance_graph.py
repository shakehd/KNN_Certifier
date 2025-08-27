from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain, combinations
from math import ceil
from pprint import pformat
from textwrap import indent
from typing import Iterable, Optional, Self, Type

import numpy as np
from scipy import optimize

from ..dataset.dataset import Dataset
from ..perturbation.perturbation import AdvRegion
from ..space.distance import Closer
from ..space.hyperplane import Hyperplane
from ..space.polyhedron import Polyhedron
from ..utils.base_types import Array1xN, Label, NDVector, VertexId

logger = logging.getLogger(__name__)


@dataclass
class Vertex:
  id: VertexId
  point: Optional[NDVector] = field(default=None)
  label: int = field(default=-1)
  closer_vertices: set[VertexId] = field(default_factory=set)  # type: ignore
  equidistant_vertices: set[VertexId] = field(default_factory=set)  # type: ignore
  edges: list[VertexId] = field(default_factory=list)  # type: ignore


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
        set(),
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
class DominanceGraph:
  vertices: dict[VertexId, Vertex]
  bisectors: dict[tuple[VertexId, VertexId], Hyperplane]
  adv_region: AdvRegion
  vertices_ids: set[VertexId] = field(init=False, default_factory=set)

  def __post_init__(self: Self) -> None:
    self.vertices_ids = set([v.id for v in self.get_vertices()])

  def __getitem__(self: Self, key: VertexId) -> Vertex:
    return self.vertices[key]

  def get_vertices(self: Self) -> list[Vertex]:
    return list(filter(lambda v: v.id != "root", self.vertices.values()))

  def get_vertices_with_same_labels(self: Self) -> dict[int, list[Vertex]]:
    vertices_with_same_labels: defaultdict[int, list[Vertex]] = defaultdict(list)

    for vertex in self.get_vertices():
      vertices_with_same_labels[vertex.label].append(vertex)

    return vertices_with_same_labels

  def _approx_max_path_length(
    self: Self,
    label: Label,
    max_length: int,
    vertices_with_same_labels: dict[Label, list[Vertex]],
  ) -> tuple[int, int]:
    vertices = list(vertices_with_same_labels.values())
    all_vertices: set[VertexId] = list(
      reduce(
        (lambda acc, val: acc.union(val.closer_vertices | set([val.id]))),  # type: ignore
        chain(*vertices),
        set(),
      )
    )  # type: ignore
    sorted_vertices: list[int] = np.array(list(sorted([int(v) for v in all_vertices])))  # type: ignore
    vertex_ids: dict[int, int] = dict((v, idx) for idx, v in enumerate(sorted_vertices))

    label_vertices_ids = [vertex_ids[vid] for vid in sorted_vertices if self[str(vid)].label == label]

    constraints: list[optimize.LinearConstraint] = []
    sizes: NDVector = np.full_like(sorted_vertices, 1)
    values: NDVector = np.full_like(sorted_vertices, 1)

    other_label_constraint: NDVector = np.full_like(sorted_vertices, 0)
    for idx, vertex_id in enumerate(sorted_vertices):
      vertex: Vertex = self[str(vertex_id)]

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

  def _get_label_vertices(self: Self, label: Label, vertices: Iterable[VertexId]) -> list[Vertex]:
    return [self[_] for _ in vertices if self[_].label == label]

  def get_neighbors_label(self: Self, k_vals: list[int], all_labels: bool = True) -> dict[int, set[int]]:
    max_k: int = max(k_vals)
    classifications: dict[int, set[int]] = dict((k, set()) for k in k_vals)

    vertices_with_same_labels: dict[Label, list[Vertex]] = self.get_vertices_with_same_labels()
    approx_upper_bounds: dict[Label, tuple[int, int]] = dict(
      [
        (
          label,
          self._approx_max_path_length(label, max_k, vertices_with_same_labels),
        )
        for label in vertices_with_same_labels
      ]
    )

    labels_to_exclude: set[Label] = set([label for label, val in approx_upper_bounds.items() if val[0] == 0])
    possible_labels_with_vertices: dict[Label, list[Vertex]] = vertices_with_same_labels
    labels = [l for l in vertices_with_same_labels if l not in labels_to_exclude]

    if len(possible_labels_with_vertices) == 1:
      for label_occur in k_vals:
        classifications[label_occur].add(self.vertices["0"].label)

      return classifications

    vertices: list[Vertex] = [v for v in self.get_vertices() if len(v.closer_vertices) < max_k]

    vertices = sorted(vertices, key=lambda val: len(val.closer_vertices))

    def get_possible_vertices(
      label: Label,
      label_counter: Counter[Label],
      existing_vertices: set[VertexId],
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
      existing_vertices: set[VertexId],
      vertex: Vertex,
      max_path_length: int,
    ) -> bool:
      all_vertices: set[VertexId] = existing_vertices | vertex.closer_vertices

      if len(all_vertices) > max_path_length:
        return True

      # other_missing_vertices = vertex.closer_vertices - existing_vertices
      # if len([_ for _ in other_missing_vertices if self[_].label != label]) > max_other_labels:
      #   return True

      lb_count = Counter(self[v].label for v in all_vertices)
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
          set(),
        )
      )  # type: ignore
      sorted_vertices: list[int] = list(sorted([int(v) for v in all_vertices]))
      vertex_ids: dict[int, int] = dict((v, idx) for idx, v in enumerate(sorted_vertices))

      label_vertices_ids = [vertex_ids[vid] for vid in sorted_vertices if self[str(vid)].label == label]

      constraints: list[optimize.LinearConstraint] = []
      sizes: NDVector = np.full_like(sorted_vertices, 1)
      values: NDVector = np.full_like(sorted_vertices, 1)

      for idx, vertex_id in enumerate(sorted_vertices):
        vertex: Vertex = self[str(vertex_id)]

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
      # constraints.append(optimize.LinearConstraint(A=sizes, lb=1, ub=capacity))
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

    for label in labels:
      if not all_labels and all([len(val) > 1 for val in classifications.values()]):
        break

      label_vertices = possible_labels_with_vertices[label]
      max_label_occur: int = approx_upper_bounds[label][1]

      other_vertices: list[Vertex] = list(
        chain(*[vertices for l, vertices in possible_labels_with_vertices.items() if l != label])
      )

      for max_length in (k for k in range(1, max_k + 1)):
        possible_label_vertices = [
          v for v in label_vertices if not ignore_vertex(label, max_label_occur, set(), v, max_length)
        ]

        exists, max_occur = exists_path(label, possible_label_vertices + other_vertices, max_length)

        if exists:
          apply_opt2 = False
          apply_opt3 = False

          try:
            init: int = min(max_length, max_occur)
            min_occur: int = ceil(max_length / len(vertices_with_same_labels.keys())) - 1
            labels_occurs: range = range(init, min_occur, -1)

            for label_occur in labels_occurs:
              possible_label_vertices = [
                v
                for v in label_vertices
                if not ignore_vertex(
                  label,
                  label_occur,
                  set(),
                  v,
                  max_length,
                )
              ]

              for label_com in combinations(possible_label_vertices, label_occur):
                all_vertices: set[VertexId] = set(
                  reduce(
                    (lambda acc, val: acc.union(val.closer_vertices) | set([val.id])),  # type: ignore
                    label_com,  # type: ignore
                    set(),
                  )  # type: ignore
                )

                label_count = Counter([self[_].label for _ in all_vertices])

                if (
                  len(all_vertices) <= max_length
                  and label_count[label] <= label_count.most_common(1)[0][1]
                  and label_count[label] == label_occur
                ):
                  if len(all_vertices) == max_length:
                    if Path.exist_valid_path(set(all_vertices), self):
                      if label_occur == max_length:
                        apply_opt3 = max_length >= ceil(max_k / 2)
                      raise StopIteration()

                  if len(all_vertices) < max_length:
                    other_possible_vertices_id = get_possible_vertices(
                      label,
                      label_count,
                      all_vertices,
                      other_vertices,
                      max_length,
                    )

                    if len(other_possible_vertices_id) > 0:
                      for other_label_com in combinations(
                        other_possible_vertices_id,
                        max_length - len(all_vertices),
                      ):
                        final_label_count = label_count + Counter([self[_].label for _ in other_label_com])

                        if final_label_count[label] >= final_label_count.most_common(1)[0][1] and Path.exist_valid_path(
                          all_vertices | set([self[_].id for _ in other_label_com]),
                          self,
                        ):
                          most_commons = final_label_count.most_common(2)
                          apply_opt2 = most_commons[0][1] - most_commons[1][1] >= max_k - max_length
                          raise StopIteration()

          except StopIteration:
            if max_length in k_vals:
              classifications[max_length].add(label)

            if apply_opt2 or apply_opt3:
              for k in (_ for _ in range(max_length, max_k + 1) if _ in k_vals):
                classifications[k].add(label)

              break

    return classifications

  @classmethod
  def build_dominance_graph(
    cls: Type[DominanceGraph],
    adv_region: AdvRegion,
    dataset: Dataset,
    max_path_length: int = 7,
  ) -> DominanceGraph:
    bisectors: dict[tuple[VertexId, VertexId], Hyperplane] = dict()
    dom_matrix: dict[VertexId, Vertex] = dict(
      [(str(i), Vertex(str(i), dataset.points[i], dataset.labels[i])) for i in range(dataset.num_points)]
    )

    initial_vertices: set[int] = set(range(dataset.num_points))
    to_remove: set[str] = set()

    for i in range(dataset.num_points):
      if len(dom_matrix[str(i)].closer_vertices) >= max_path_length:
        initial_vertices.discard(i)
        to_remove.add(str(i))
        continue

      for j in range(i + 1, dataset.num_points):
        if len(dom_matrix[str(j)].closer_vertices) >= max_path_length:
          continue

        bisectors[(str(i), str(j))] = Hyperplane.build_equidistant_plane(dataset.points[i], dataset.points[j])

        match adv_region.get_closer(bisectors[(str(i), str(j))]):
          case Closer.FIRST:
            dom_matrix[str(i)].edges.append(str(j))
            dom_matrix[str(j)].closer_vertices.add(str(i))
            initial_vertices.discard(j)

          case Closer.SECOND:
            dom_matrix[str(j)].edges.append(str(i))
            dom_matrix[str(i)].closer_vertices.add(str(j))
            initial_vertices.discard(i)

          case Closer.BOTH:
            dom_matrix[str(i)].edges.append(str(j))
            dom_matrix[str(i)].equidistant_vertices.add(str(j))

            dom_matrix[str(j)].edges.append(str(i))
            dom_matrix[str(j)].equidistant_vertices.add(str(i))

    root_edges: list[VertexId] = [str(i) for i in initial_vertices]
    dom_matrix["root"] = Vertex("root", edges=root_edges)

    for node in to_remove:
      del dom_matrix[node]

    for vertex in dom_matrix.values():
      vertex.closer_vertices -= to_remove
      vertex.equidistant_vertices -= to_remove
      vertex.edges = [edge for edge in vertex.edges if edge not in to_remove]

    logger.debug("\t dominance graph: \n")
    logger.debug("%s\n", indent(pformat(dom_matrix, compact=True), "\t\t"))
    return cls(dom_matrix, bisectors, adv_region)

from __future__ import annotations
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from heapq import heapify, heappop, heappush
from typing import Optional, Self, Type
from pprint import pformat
from textwrap import indent
from itertools import repeat, chain
from functools import reduce
from math import ceil
import numpy as np
from scipy import optimize

from ..perturbation.perturbation import AdvRegion
from ..space.hyperplane import Hyperplane
from ..space.polyhedron import Polyhedron

from ..dataset.dataset import Dataset
from ..space.distance import Closer
from ..utils.base_types import  Array1xN, NDVector

logger = logging.getLogger(__name__)

VertexId = str
Label = int

@dataclass
class Vertex:
  id: VertexId
  point: Optional[NDVector] = field(default=None)
  label: int = field(default=-1)
  closer_vertices: set[VertexId] = field(default_factory=set)
  equidistant_vertices: set[VertexId] = field(default_factory=set)
  edges: list[VertexId] = field(default_factory=list)

  def copy(self: Self) -> Vertex:
    return Vertex(
      self.id,
      self.point,
      self.label,
      self.closer_vertices.copy(),
      self.equidistant_vertices.copy(),
      self.edges.copy())

  def add_to_closer(self: Self, vertex_id: VertexId):
    self.equidistant_vertices.discard(vertex_id)
    self.edges = [adj for adj in self.edges if adj != vertex_id]
    self.closer_vertices.add(vertex_id)


@dataclass(order=True)
class PrioritizedItem:
    priority: tuple[int, int]
    item: Path =field(compare=False)

class Safe_Vertex_Error(Enum):
  MISSING_ANCESTOR = auto()
  CIRCULAR_PATH    = auto()
  UNSATISFIED_LP   = auto()
  NONE             = auto()

@dataclass
class Path:
  vertices: list[VertexId] = field(default_factory=list)
  polyhedron: Polyhedron = field(default_factory=Polyhedron.perturbation)
  label_occurrences: Counter[int] = field(default_factory=Counter)
  max_length: int = field(default_factory=int)
  min_length: int = field(default=1)

  @property
  def last(self: Self) -> VertexId:
    return self.vertices[-1]

  @property
  def most_common_label(self: Self) -> tuple[int, int]:
    return self.label_occurrences.most_common(1)[0]

  @property
  def most_common_labels(self: Self) -> dict[Label, int]:
    max_freq: int = self.most_common_label[1]
    most_freq_labels: list[tuple[Label, int]] = [(l, c) for l,c in self.label_occurrences.items()\
                                            if c == max_freq]
    return dict(most_freq_labels)

  def len(self: Self) -> int:
    return len(self.vertices)

  def is_safe(self: Self, vertex: Vertex,
              dom_graph : DominanceGraph) -> tuple[Safe_Vertex_Error, None | Polyhedron]:

    def build_new_polyhedrons() -> Polyhedron:
      not_ancestor = set(self.vertices) - vertex.closer_vertices

      inequalities_lhs: list[Array1xN] = []
      inequalities_rhs: list[float] = []

      equidistant_vertices: set[VertexId] = \
            vertex.equidistant_vertices - set(self.vertices)

      equidistant_vertices = set(
        [v_id for v_id in equidistant_vertices
         if len(dom_graph[v_id].closer_vertices) <= self.len()
        ]
      )

      if not not_ancestor and not equidistant_vertices:
        return self.polyhedron.copy()

      bisectors = dom_graph.bisectors
      if equidistant_vertices:
        for v in equidistant_vertices:
          if (vertex.id, v) in bisectors:
            bisector = bisectors[(vertex.id, v)]
            inequalities_lhs.append(bisector.coefficients)
            inequalities_rhs.append(bisector.constant)
          else:
            bisector = bisectors[(v, vertex.id)]
            inequalities_lhs.append(-bisector.coefficients)
            inequalities_rhs.append(-bisector.constant)

      if not_ancestor:
        for v in not_ancestor:
          if (v, vertex.id) in bisectors:
            bisector = bisectors[(v, vertex.id)]
            inequalities_lhs.append(bisector.coefficients)
            inequalities_rhs.append(bisector.constant)
          else:
            bisector = bisectors[(vertex.id, v)]
            inequalities_lhs.append(-bisector.coefficients)
            inequalities_rhs.append(-bisector.constant)

      return self.polyhedron.refine(inequalities_lhs, inequalities_rhs) # type: ignore

    missing_ancestor: bool = bool(vertex.closer_vertices) and \
                                  (not vertex.closer_vertices <= set(self.vertices))
    circular_path: bool = vertex.id in self.vertices

    if missing_ancestor:
      return Safe_Vertex_Error.MISSING_ANCESTOR, None

    if circular_path:
      return Safe_Vertex_Error.CIRCULAR_PATH, None

    new_polyhedron = build_new_polyhedrons()

    if not new_polyhedron.is_valid():
      return Safe_Vertex_Error.UNSATISFIED_LP, None

    return Safe_Vertex_Error.NONE, new_polyhedron

  def add_vertex(self: Self, vertex: Vertex,
                  dom_graph : DominanceGraph)  -> Safe_Vertex_Error | Path:

    safe_error, polyhedron = self.is_safe(vertex, dom_graph)

    if safe_error != Safe_Vertex_Error.NONE:
      return safe_error

    assert polyhedron is not None
    return Path(
      self.vertices + [vertex.id],
      polyhedron,
      self.label_occurrences + Counter([vertex.label]),
      self.max_length,
      self.min_length
    )

  def __len__(self: Self) -> int:
    return len(self.vertices)

  def __eq__(self: Self, other: object) -> bool:

    if not isinstance(other, Path):
      return False

    return self.vertices == other.vertices

  def __str__(self) -> str:
    return '[' + ', '.join(self.vertices) + ']'


  @classmethod
  def emptyPath(cls:  type[Path], max_length: int=0) -> Path:
    return cls(max_length=max_length)
  @classmethod
  def singlePath(cls:  type[Path], vertexId: VertexId) -> Path:
    return cls([vertexId])

  @classmethod
  def check_path(cls: type[Path],
                 init_path: Path,
                 vertices: list[VertexId],
                 bisectors: dict[tuple[VertexId, VertexId], Hyperplane],
                 dom_graph : DominanceGraph,
      ) -> list[bool]:

    curr_path: Path = init_path
    valid_lengths = list(repeat(False, len(vertices)))
    for ix, vx in enumerate(vertices):
      res = curr_path.add_vertex(dom_graph[vx], dom_graph)

      if isinstance(res, Path):
        valid_lengths[ix] = True
        curr_path = res
      else:
        break

    return valid_lengths

@dataclass
class DominanceGraph:
  vertices: dict[VertexId, Vertex]
  bisectors: dict[tuple[VertexId, VertexId], Hyperplane]
  vertices_ids: set[VertexId] = field(init=False, default_factory=set)

  def __post_init__(self: Self) -> None:
    self.vertices_ids = set([v.id for v in self.get_vertices()])

  def __getitem__(self: Self, key: VertexId) -> Vertex:
    return self.vertices[key]

  def get_vertices(self: Self) -> list[Vertex]:
    return list(filter(lambda v: v.id != 'root', self.vertices.values()))

  def _get_vertices_with_same_labels(self: Self) -> dict[int, list[Vertex]]:

    vertices_with_same_labels: defaultdict[int, list[Vertex]] = defaultdict(list)

    for vertex in self.get_vertices():
      vertices_with_same_labels[vertex.label].append(vertex)

    return vertices_with_same_labels

  def _get_labels_occurrences(self: Self) -> dict[Label, int]:

    labels_occurrences: defaultdict[int, int] = defaultdict(int)

    for vertex in self.get_vertices():
      labels_occurrences[vertex.label] += 1

    return labels_occurrences

  def _get_possible_label_occurrences(self: Self, max_k_value: int,
                                      labels_to_exclude: set[Label]) -> dict[Label, list[Vertex]]:

    labels_vertices: defaultdict[int, list[Vertex]] = defaultdict(list)

    label_count: Counter[int] = Counter([
      v.label for v in self.get_vertices() if len(v.closer_vertices) < max_k_value
    ])

    for vx in self.get_vertices():

      if vx.label not in labels_to_exclude:
        lb_count = Counter([self[v].label for v in vx.closer_vertices])
        another_label_majority = False
        if len(lb_count) > 0:
          most_common = lb_count.most_common(1)[0]
          another_label_majority = most_common[1] >= ceil(max_k_value/2) and\
                                  most_common[0] != vx.label

        if another_label_majority:
          continue

        if label_count[vx.label] >= (len(vx.closer_vertices) + 1)//2:
          labels_vertices[vx.label].append(vx)

    return labels_vertices

  def _approx_max_path_length(self: Self, vertices: list[Vertex], max_length: int,
                              vertices_with_same_labels: dict[Label, list[Vertex]],
                              init_path: Path | None = None) -> int:

      label: Label = vertices[0].label
      vertex_added: int = 0
      all_vertices: set[VertexId] = list(reduce((lambda acc, val: acc.union(val.closer_vertices | set([val.id]))), # type: ignore
                                            vertices, set())) # type: ignore
      sorted_vertices: list[int] = np.array(list(sorted([int(v) for v in all_vertices]))) # type: ignore
      vertex_ids: dict[int, int] = dict((v, idx) for idx, v in enumerate(sorted_vertices))

      label_vertices_ids = [vertex_ids[vid] for vid in sorted_vertices if self[str(vid)].label == label]

      constraints: list[optimize.LinearConstraint] = []
      sizes: NDVector = np.full_like(sorted_vertices, 1)
      values: NDVector = np.full_like(sorted_vertices, 0)

      other_label_constraint: NDVector = np.full_like(sorted_vertices, 0)
      for idx, vertex_id in enumerate(sorted_vertices):
        vertex: Vertex = self[str(vertex_id)]

        if vertex.label == label:

          values[idx] = 1

        else:

          other_label_constraint[idx] = 1


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
          other_label_constraint[[vertex_ids[int(v.id)] for v in vertices_with_same_labels[alabel] if int(v.id) in vertex_ids]] = -1

          constraints.append(optimize.LinearConstraint(A=other_label_constraint, lb=0, ub=len(label_vertices_ids)))

      if init_path is not None:
        constraint: NDVector = np.full_like(sorted_vertices, 0)
        constraint[[vertex_ids[int(vid)] for vid in init_path.vertices]] = 1
        constraints.append(optimize.LinearConstraint(A=constraint, lb=0, ub=len(init_path)))

      bounds = optimize.Bounds(0, 1)
      integrality = np.full_like(sorted_vertices, True)
      capacity: int = max_length
      constraints.append(optimize.LinearConstraint(A=sizes, lb=0, ub=capacity))

      solution: optimize.OptimizeResult = optimize.milp(c=-values, constraints=constraints,\
                               integrality=integrality, bounds=bounds) # type: ignore

      vertex_added: int = values[solution.x != 0].sum() # type: ignore

      if vertex_added != 0:

        return min(sum([
                min(vertex_added, len(vxs)) for vxs in vertices_with_same_labels.values() # type: ignore
              ]), max_length)

      return 0

  def get_neighbors_label(self: Self, k_vals: list[int]) -> dict[int, set[int]]:

    max_k: int = max(k_vals)
    classifications: dict[int, set[int]] = defaultdict(set)

    vertices_with_same_labels: dict[Label, list[Vertex]] = self._get_vertices_with_same_labels()
    max_approx_lengths: dict[Label, int] = dict([
      (label, self._approx_max_path_length(vertices_with_same_labels[label], max_k, vertices_with_same_labels))
      for label in vertices_with_same_labels
    ])
    labels_to_exclude: set[Label] = set([label for label, val in max_approx_lengths.items() if val == 0])
    possible_labels_with_vertices: dict[Label, list[Vertex]] = vertices_with_same_labels

    if len(possible_labels_with_vertices) == 1:

      for k in k_vals:
        classifications[k].add(self.vertices['0'].label)

      return classifications

    vertices: list[Vertex] = [
      v for v in self.get_vertices() if len(v.closer_vertices) < max_k
    ]

    vertices = sorted(vertices, key=lambda val: len(val.closer_vertices))

    k_classified_with: dict[int, list[bool]] = dict([
      (label, list(repeat(False, max_k))) for label in possible_labels_with_vertices
    ])

    def get_possible_vertices(path: Path, max_length: int,
                              label_vertices: list[Vertex],
                              other_vertices: list[Vertex]) -> list[Vertex]:


      possible_vertices: list[Vertex] = []
      label_vertices_count = len(label_vertices)
      for vertex in other_vertices:
        if vertex.id not in path.vertices and not ignore_vertex(path, vertex, label_vertices_count, max_length):

          possible_vertices.append(vertex)

      label_vertices_to_include = vertex_to_include(path, label_vertices, possible_vertices)
      possible_vertices.extend(label_vertices_to_include)

      return possible_vertices


    def approx_max_path_length(label: Label, path: Path, vertices: list[Vertex]) -> int:

      all_vertices: set[VertexId] = set(reduce((lambda acc, val: acc.union(val.closer_vertices | set([val.id]))), # type: ignore
                                            vertices, set())) # type: ignore
      sorted_vertices: list[int] = list(sorted([int(v) for v in all_vertices | set(path.vertices)]))
      vertex_ids: dict[int, int] = dict((v, idx) for idx, v in enumerate(sorted_vertices))

      label_vertices_ids = [vertex_ids[vid] for vid in sorted_vertices if self[str(vid)].label == label]
      path_vertex_ids = [vertex_ids[int(vid)] for vid in path.vertices]

      constraints: list[optimize.LinearConstraint] = []
      sizes: NDVector = np.full_like(sorted_vertices, 1)
      values: NDVector = np.full_like(sorted_vertices, 1)

      for idx, vertex_id in enumerate(sorted_vertices):
        vertex: Vertex = self[str(vertex_id)]

        if len(vertex.closer_vertices) > 0:
          for closer_id in vertex.closer_vertices:
            constraint: NDVector = np.full_like(sorted_vertices, 0)
            constraint[vertex_ids[int(closer_id)]] = 1
            constraint[idx] = -1
            constraints.append(optimize.LinearConstraint(A=constraint, lb=0, ub=1))

      constraint: NDVector = np.full_like(sorted_vertices, 0)
      constraint[path_vertex_ids] = 1
      constraints.append(optimize.LinearConstraint(A=constraint, lb=len(path), ub=len(path)))

      for other_label in possible_labels_with_vertices:
        constraint: NDVector = np.full_like(sorted_vertices, 0)
        constraint[label_vertices_ids] = 1
        if other_label != label:
          constraint[[vertex_ids[int(v.id)] for v in possible_labels_with_vertices[other_label]
                                                  if int(v.id) in vertex_ids]] = -1
          constraints.append(optimize.LinearConstraint(A=constraint, lb=0, ub=len(label_vertices_ids)))

      constraint: NDVector = np.full_like(sorted_vertices, 1)
      constraint[label_vertices_ids] = 0
      constraint[path_vertex_ids] = 0
      constraints.append(optimize.LinearConstraint(A=constraint, lb=1, ub=np.inf))

      bounds = optimize.Bounds(0, 1)
      integrality = np.full_like(sorted_vertices, True)
      capacity: int = max_k
      constraints.append(optimize.LinearConstraint(A=sizes, lb=0, ub=capacity))

      solution = optimize.milp(c=-values, constraints=constraints,\
                              integrality=integrality, bounds=bounds)

      if solution.success:
        sol = [vid for vid in sorted_vertices if solution.x[vertex_ids[int(vid)]] != 0]

        return len(sol)
      else:
        return len(path)

    def vertex_to_include(init_path: Path, label_vertices: list[Vertex],
                          other_vertices: list[Vertex]) -> list[Vertex]:
      vertex_to_include: list[Vertex] = []

      label = label_vertices[0].label
      max_length = max_approx_lengths[label]
      bisectors = self.bisectors

      for vertex in label_vertices:

        vertex_cp: Vertex = vertex.copy()

        to_include: bool = True

        if vertex.id not in init_path.vertices and\
          len(other_vertices) > 0:

            for other_vertex in other_vertices:
              if ignore_vertex(init_path, vertex_cp, len(label_vertices), max_length):
                to_include = False
                break

              if vertex.id != other_vertex.id:
                inequalities_lhs: list[Array1xN] = []
                inequalities_rhs: list[float] = []
                if (vertex.id, other_vertex.id) in bisectors:
                  bisector = bisectors[(vertex.id, other_vertex.id)]
                  inequalities_lhs.append(bisector.coefficients)
                  inequalities_rhs.append(bisector.constant)#
                else:
                  bisector = bisectors[(other_vertex.id, vertex.id)]
                  inequalities_lhs.append(-bisector.coefficients)
                  inequalities_rhs.append(-bisector.constant)

                if not init_path.polyhedron.refine(inequalities_lhs, inequalities_rhs).is_valid(): # type: ignore
                  vertex_cp.add_to_closer(other_vertex.id)

        else:
          to_include = False

        if to_include:
          vertex_to_include.append(vertex_cp)

      return vertex_to_include

    def applied_optimisation_on_length(path: Path,
                                       priority_queue: list[PrioritizedItem],
                                       max_length: int) -> tuple[bool, int]:

      path_length = len(path)
      most_commons = path.label_occurrences.most_common(2)


      if len(most_commons) == 1 and label in path.most_common_labels:

        for k in range(max_k+1):
          if k in k_vals:
            k_classified_with[label][k-1] = True
            classifications[k].add(label)

        priority_queue = []
        heapify(priority_queue)
        return True, 0

      else:

        if len(most_commons) == 1 or \
          most_commons[0][1] - most_commons[1][1] >= max_k - path_length:

          for k in range(path_length, max_k+1):

            if k in k_vals:
              k_classified_with[most_commons[0][0]][k-1] = True
              classifications[k].add(most_commons[0][0])

          if most_commons[0][0] == label:
            k_with_missing_label: list[int] = [ix for ix, val in enumerate(k_classified_with[label]) if not val]

            max_path_length = 0 if not k_with_missing_label else min(k_with_missing_label[-1] + 1, max_length)
            priority_queue = [item for item in priority_queue if abs(item.priority[1]) <= max_path_length]

            heapify(priority_queue)
            return True, max_path_length

      return False, max_length

    def ignore_vertex(path: Path, vertex: Vertex, label_count: int,
                      max_path_length: int) -> bool:

      potential_path = set(path.vertices) | vertex.closer_vertices


      lb_count = Counter(self[v].label for v in potential_path)
      lb_count.update([vertex.label])
      most_common = lb_count.most_common(1)[0]

      insufficient_vertex: bool = most_common[1] - lb_count[label] > \
                            label_count - lb_count[label]

      insufficient_length = most_common[1] - lb_count[label] > \
                            max_path_length - (len(potential_path)+1)


      if insufficient_vertex or insufficient_length:
        return True

      return False

    def extend_path(path: Path, vertex: Vertex,
                    label: Label, queue: list[PrioritizedItem],
                    distinct_paths: defaultdict[int, set[tuple[VertexId,...]]]) -> bool:

      path_length: int = len(path)
      if tuple(path.vertices + [vertex.id]) not in distinct_paths[path_length+1]:

        result = path.add_vertex(vertex, self)

        if result == Safe_Vertex_Error.UNSATISFIED_LP or isinstance(result, Path):

          if isinstance(result, Path):

            labels_priority = -result.label_occurrences[label]
            item: PrioritizedItem = PrioritizedItem(
              (labels_priority, -1*len(result)),
              result
            )
            distinct_paths[len(result)].add(tuple(result.vertices))
            heappush(queue, item)
            return True

        if result == Safe_Vertex_Error.MISSING_ANCESTOR:

          missing_ancestor =  vertex.closer_vertices - set(path.vertices)


          for ancestor in vertex.closer_vertices - set(path.vertices):

            extend_path(path, self[ancestor], label, queue, distinct_paths)

          return False

      return False

    def build_init_paths(vertices: list[Vertex], label: Label,
                         max_length: int,
                         distinct_paths: defaultdict[int, set[tuple[VertexId,...]]])  -> list[Path]:

      # init_paths: list[Path] = [Path.emptyPath()]
      init_paths: list[Path] = []

      priority_queue: list[PrioritizedItem] = []
      heappush(priority_queue, PrioritizedItem((0, 0), Path.emptyPath()))

      label_count = len(vertices)

      label_vertices = sorted(vertices, key=lambda v: len(v.closer_vertices))

      max_path_length: int = max_length
      while len(priority_queue):

        queue_item: PrioritizedItem = heappop(priority_queue)
        path: Path = queue_item.item
        path_length: int = len(path)

        # print(f'Building path {path}')

        if all(k_classified_with[label][:max_path_length]):
          return []

        if path_length > 0:
          vertex = self[path.last]
          if len(vertex.closer_vertices) > 0 and path.label_occurrences[label] == 1:
            labels_count: Counter[int] = Counter([self[v].label for v in vertex.closer_vertices])
            most_commons = labels_count.most_common(2)
            if most_commons[0][0] != label:
              path.min_length = 2*most_commons[0][1]
            elif len(most_commons) > 1:
              path.min_length = 2*most_commons[1][1]

          if all(k_classified_with[label][path_length-1:]) or\
            all(k_classified_with[label][path.min_length-1:]):
            continue

          if path_length >= ceil(max_k/2):

            applied, path_max_length = applied_optimisation_on_length(path, priority_queue, max_path_length)
            max_path_length = path_max_length
            if applied:
                continue

          if path_length > 0:

            most_commons = path.label_occurrences.most_common(2)
            if len(most_commons) == 1 and most_commons[0][0] == label:

              for k in range(path_length+1):
                if k in k_vals:
                  k_classified_with[label][k-1] = True
                  classifications[k].add(label)

            else:

              most_common_labels = path.most_common_labels

              if path_length in k_vals and label in most_common_labels:
                k_classified_with[label][path_length-1] = True
                classifications[path_length].add(label)

              if path_length == max_path_length:

                k_with_missing_label: list[int] = [ix for ix, val in enumerate(k_classified_with[label])
                                                      if not val]
                max_path_length = 0 if not k_with_missing_label else min(k_with_missing_label[-1] + 1, max_length)
                priority_queue = [item for item in priority_queue if abs(item.priority[1]) <= max_path_length]

                heapify(priority_queue)

                continue


          if all(k_classified_with[label][:max_path_length]):
            return []

        path_extended: bool = False
        if path_length <= max_path_length:

          for vertex in filter(lambda v: v.id not in path.vertices, label_vertices):

            # if not ignore_vertex(path, vertex, label_count, max_length):

              path_extended |= extend_path(path, vertex, label, priority_queue, distinct_paths)

        if not path_extended and 0 < path_length <= max_path_length\
           and path.most_common_label[1] <= path.label_occurrences[label]\
           and path not in init_paths:          #  and self[path.vertices[-1]].label == label\

            init_paths.append(path)

      return init_paths

    def extend_init_path(label: Label, init_path: Path, possible_vertices: list[Vertex],
                         max_length: int,
                         distinct_paths: defaultdict[int, set[tuple[VertexId,...]]]) -> None:

      init_path_length = len(init_path)
      max_label_occur = len([_ for _ in possible_vertices if _.label == label]) + init_path.label_occurrences[label]
      max_path_length = max_length

      priority_queue: list[PrioritizedItem] = []
      heappush(
        priority_queue,
        PrioritizedItem((0, -init_path_length), init_path)
      )

      while len(priority_queue):

        path: Path = heappop(priority_queue).item
        path_length: int = len(path)

        if path_length > 0:
          if all(k_classified_with[label][init_path.min_length:]) or\
            all(k_classified_with[label][path_length-1:]):
            continue

          if path_length >= ceil(max_k/2):
            most_commons = path.label_occurrences.most_common(2)

            if len(most_commons) == 1 or \
              most_commons[0][1] - most_commons[1][1] > max_k - path_length:

              for k in range(path_length, max_k+1):

                if k in k_vals:
                  k_classified_with[most_commons[0][0]][k-1] = True

              if most_commons[0][0] == label:
                k_with_missing_label: list[int] = [ix for ix, val in enumerate(k_classified_with[label])
                                                      if not val]
                new_max_path_length = 0 if not k_with_missing_label else k_with_missing_label[-1] + 1
                max_path_length = min(new_max_path_length, max_path_length)
                priority_queue = [item for item in priority_queue
                                        if abs(item.priority[1]) <= max_path_length]

                heapify(priority_queue)

              continue


          max_freq: int = path.label_occurrences.most_common(1)[0][1] # type: ignore
          most_freq_labels: list[int] = [k for k,c in path.label_occurrences.items()\
                                          if c == max_freq]

          if label in most_freq_labels:
            k_classified_with[label][path_length-1] = True

            if path_length == max_path_length:
              k_with_missing_label: list[int] = [ix for ix, val in enumerate(k_classified_with[label])
                                                    if not val]
              max_path_length = 0 if not k_with_missing_label else min(k_with_missing_label[-1] + 1, max_path_length)
              priority_queue = [item for item in priority_queue
                                      if abs(item.priority[1]) <= max_path_length]

              heapify(priority_queue)

          if all(k_classified_with[label][:max_path_length]):
            break


        if path_length == 0 or (path.most_common_label[1] <= max_label_occur and path_length < max_path_length):

          for adj in possible_vertices:
            if not ignore_vertex(path, adj, max_label_occur, max_length):
              if adj.label == label:
                extend_path(path, adj, label, priority_queue, distinct_paths)

              if adj.label != label and \
                max_label_occur >= path.label_occurrences[adj.label] + 1 and\
                not all(k_classified_with[label][init_path.min_length-1:max_path_length]):

                extend_path(path, adj, label, priority_queue, distinct_paths)

    for label in possible_labels_with_vertices:

      if label not in labels_to_exclude:

        label_vertices = possible_labels_with_vertices[label]
        min_path_length = min([len(v.closer_vertices) for v in label_vertices])
        distinct_paths: defaultdict[int, set[tuple[VertexId,...]]] = defaultdict(set)

        k_classified_with[label][:min_path_length] = list(repeat(True, min_path_length))

        other_vertices: list[Vertex] = list(chain(*[vertices for l, vertices in possible_labels_with_vertices.items()
                                                                          if l != label]))

        for i in range(max_k):
          if i+1 not in k_vals:
            k_classified_with[label][i] = True

        init_paths = build_init_paths(possible_labels_with_vertices[label],
                                      label, max_approx_lengths[label], distinct_paths)

        if len(init_paths) == 0 or all(k_classified_with[label]):
          continue

        sorted_paths = list(sorted(init_paths, key=lambda p: len(p.vertices), reverse=True))

        for init_path in sorted_paths:

          possible_vertices = get_possible_vertices(init_path, max_approx_lengths[label], possible_labels_with_vertices[label], other_vertices)

          path_max_length = approx_max_path_length(label, init_path, possible_vertices)

          if len(init_path) > 0:
            if all(k_classified_with[label][init_path.min_length-1:path_max_length]) or\
              all(k_classified_with[label][len(init_path)-1:path_max_length]):
              continue

            path_length = len(init_path)
            if path_length in k_vals:
              k_classified_with[label][path_length-1] = True

            if init_path.most_common_label[1] == path_length:
              for k in range(1, path_length+1):
                if k in k_vals:
                  k_classified_with[label][k-1] = True


          extend_init_path(label, init_path, possible_vertices, path_max_length, distinct_paths)

        for k in range(min_path_length+1,max_k +1):

          if k in k_vals and k_classified_with[label][k-1]:
            classifications[k].add(label)

    return classifications

  @classmethod
  def build_dominance_graph(cls: Type[DominanceGraph],
                            adv_region: AdvRegion,
                            dataset: Dataset,
                            max_path_length: int = 7) -> DominanceGraph:

    bisectors: dict[tuple[VertexId, VertexId], Hyperplane] = dict()
    dom_matrix: dict[VertexId, Vertex] = dict(
      [(str(i), Vertex(str(i), dataset.points[i],  dataset.labels[i]))
                                            for i in range(dataset.num_points)]
    )

    initial_vertices: set[int] = set(range(dataset.num_points))
    to_remove: set[str] = set()

    for i in range(dataset.num_points):

      if len(dom_matrix[str(i)].closer_vertices) >= max_path_length:
        initial_vertices.discard(i)
        to_remove.add(str(i))
        continue

      for j in range(i+1, dataset.num_points):

        if len(dom_matrix[str(j)].closer_vertices) >= max_path_length:
          continue

        bisectors[(str(i), str(j))] = Hyperplane.build_equidistant_plane(
          dataset.points[i],
          dataset.points[j]
        )

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

    root_edges: list[VertexId]= [str(i) for i in initial_vertices]
    dom_matrix['root'] = Vertex('root', edges=root_edges)

    for node in to_remove:
      del dom_matrix[node]

    for vertex in dom_matrix.values():
      vertex.closer_vertices -= to_remove
      vertex.equidistant_vertices -= to_remove
      vertex.edges = [edge for edge in vertex.edges if edge not in to_remove]

    logger.debug("\t dominance graph: \n")
    logger.debug('%s\n', indent(pformat(dom_matrix, compact=True),'\t\t'))
    return cls(dom_matrix, bisectors)
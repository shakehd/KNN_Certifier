from collections.abc import Iterable, Iterator


def permutations_with_constraints[T](
  elements: Iterable[tuple[set[T], T]], size: int, invalid_prefixes: set[tuple[T, ...]]
) -> Iterator[tuple[T, ...]]:
  """
  Generate permutations of elements with constraints on the selection.

  Args:
      elements: An iterable of tuples, where each tuple contains a set of constraints and an element.
      size: The size of the permutations to generate.

  Yields:
      Tuples representing valid permutations of the specified size.
  """

  def _go(elems: Iterable[tuple[set[T], T]], size: int, selected: tuple[T, ...]) -> Iterator[tuple[T, ...]]:
    if selected in invalid_prefixes:
      return []  # type: ignore

    if size == 1:
      for elem in (_ for _ in elems if _[1] not in selected):
        yield (elem[1],)
    else:
      for elem in (_ for _ in elems if _[1] not in selected):
        if set(selected).issuperset(elem[0]):
          for perm in _go(elems, size - 1, selected + (elem[1],)):
            yield (elem[1],) + perm

  for permutation in _go(elements, size, ()):
    yield permutation

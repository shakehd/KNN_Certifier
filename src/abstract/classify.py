
from dataclasses import dataclass
import logging
import time
from numpy import zeros

from src.abstract.abstract_classifier import AbstractClassifier
from src.dataset.dataset import DatasetProps
from src.perturbation.perturbation import Perturbation
from src.utils.configuration import Configuration

from ..utils.base_types import ArrayNxM, NDVector

logger = logging.getLogger(__name__)


@dataclass
class Result:
  classification: list[str]
  robustness: list[str]
  robustness_count: NDVector
  stability: list[str]
  stability_count: NDVector

def classify_point(abstract_classifier: AbstractClassifier, num_point:int, test_point: ArrayNxM, test_label:int,
                   params: Configuration, dataset_props: DatasetProps, all_labels: bool = True) -> Result:
  try:

    AbstractClassifier.point_number = num_point

    epsilon: float = params['abstraction']['epsilon']
    k_values: list[int] = params['knn_params']['k_values']

    stable_count = zeros(len(k_values))
    robust_count = zeros(len(k_values))

    logger.info(f"-- Classifying point {num_point} {test_point} with label {test_label} --\n")

    perturbation = Perturbation(test_point, epsilon, dataset_props)
    clock_st: float = time.time()
    labels = abstract_classifier.get_classification(perturbation, k_values, all_labels)
    elapsed_clock: float = time.time() - clock_st
    logger.info(f"Classification time: {elapsed_clock}")
    classification_result: list[str]  = [str(num_point), str(test_label)]
    robustness_result: list[str]      = [str(num_point)]
    stability_result: list[str]       = [str(num_point)]

    logger.info('\tFinal labels: ')
    for ix, (k, classification) in enumerate(labels.items()):

      logger.info(f'\t\tk = {k} -> {classification}', )
      classification_result.append(str(classification))

      is_stable = len(classification) == 1
      is_robust = is_stable and classification.pop() == test_label

      robustness_result.append('Yes' if is_robust else 'No')
      stability_result.append('Yes' if is_stable else 'No')

      stable_count[ix] += is_stable
      robust_count[ix] += is_robust

    logger.info('\n')
    return Result(classification_result,
                  robustness_result,
                  robust_count,
                  stability_result,
                  stable_count)

  except Exception:
    logger.exception(f'An error occurred while classifying: ')
    raise

def classify_point_async(args: tuple[AbstractClassifier, int, ArrayNxM, int, Configuration, DatasetProps]) -> Result:
    return classify_point(*args)


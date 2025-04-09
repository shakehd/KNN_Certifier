from concurrent.futures import Future, as_completed
import csv
from multiprocessing import get_context
from typing import Optional
from os.path import exists, join
from os import mkdir
from tqdm import tqdm
from numpy import zeros
import logging
import argparse
import time
from pebble import ProcessExpired, ProcessPool

from src.abstract.abstract_classifier import AbstractClassifier
from src.dataset.loader import DataLoader
from src.logging.logger import ProcessLogger
from src.perturbation.perturbation import Perturbation
from src.utils.base_types import ArrayNxM, NDVector
from src.utils.configuration import Configuration
from src.abstract.classify import Result, classify_point
from src.utils.base_types import NDVector

logger = logging.getLogger(__name__)

def get_args_parser() -> argparse.ArgumentParser:

  parser = argparse.ArgumentParser(description='Abstract Knn')
  parser.add_argument('config_filepath', metavar='CONFIGFILE',
                      help='name of the config file to read (it must be inside the config folder !!).')
  parser.add_argument('--random-state', metavar='RANDOM', type=int,
                      help='random seed used when partitioning the dataset.')
  parser.add_argument('--all-labels',
                      help='Compute all labels. (default False)',
                      action='store_true')
  parser.add_argument('--partition-size', metavar='PSIZE', type=int,
                      default=100,
                      help='maximum number of data points in a partition (default 20).')
  parser.add_argument('--no-parallel', action='store_true',
                      help='Classify point sequentially.')
  parser.add_argument('--log ', dest='log_level',
                      choices=['INFO', 'DEBUG', 'ERROR'],
                      default='ERROR',type = str.upper,
                      help='log level used during the verification phase (default ERROR).')

  return parser


def parallel_main(params: Configuration, level:str, partition_size: int = 20,
                  random_state: Optional[int] = None, all_labels: bool = True) -> None:

  global abstract_classifier
  k_values = params['knn_params']['k_values']

  stable_count: NDVector = zeros(len(k_values))
  robust_count: NDVector = zeros(len(k_values))

  loader = DataLoader(params['dataset'])

  training_set, test_set, dataset_props = loader.load_datasets(params['base_dirs']['dataset'])
  abstract_classifier.fit(training_set, partition_size, random_state)

  classification_results: list[list[str]] = []
  robustness_results: list[list[str]] = []
  stability_results: list[list[str]] = []

  clock_st: float = time.time()

  points: list[tuple[ArrayNxM, int]] = list(zip(test_set.points, test_set.labels))
  tot_points: int = len(points)
  classified_points: int = 0

  timeout_tasks : list[int] = []
  timeout: int = 6 * 60 * 60 # six hours
  with tqdm(total=tot_points,
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_inv_fmt} {postfix}]',
             desc='Verifying') as pbar:

    pool: ProcessPool
    with ProcessPool(
      context=get_context('fork'),
      initializer=ProcessLogger.configure_logger_for_process,
      initargs=[ProcessLogger.get_logger().queue, ProcessLogger.get_logger().level],
      max_tasks=50
      ) as pool:

      results: dict[Future[Result], int] =\
          {pool.schedule(classify_point, args=(abstract_classifier, ix+1, point, label, params, dataset_props, all_labels), timeout=timeout):ix
              for ix, (point, label) in enumerate(points)}

      for result in as_completed(results):

        try:
          classification_result = result.result()
          classification_results.append(classification_result.classification)
          robustness_results.append(classification_result.robustness)
          stability_results.append(classification_result.stability)
          stable_count += classification_result.stability_count
          robust_count += classification_result.robustness_count
          classified_points += 1
          pbar.set_postfix_str('ROB={}%, STAB={}%'.format(
            round(sum(robust_count) / (classified_points * len(k_values)) * 100, 1),
            round(sum(stable_count) / (classified_points * len(k_values)) * 100, 1)
          ))
          pbar.update(1)
        except StopIteration:
          break
        except TimeoutError as error:
          print(F"Classification of point {results[result]} took longer than {error.args[1]} seconds")
          timeout_tasks.append(results[result])
        except ProcessExpired as error:
          print("%s. Exit code: %d" % (error, error.exitcode))
          timeout_tasks.append(results[result])
        except Exception as error:
          print("function raised %s" % error)
          print(error.traceback)  # Python's traceback of remote process
          timeout_tasks.append(results[result])

    with ProcessPool(
      context=get_context('fork'),
      initializer=ProcessLogger.configure_logger_for_process,
      initargs=[ProcessLogger.get_logger().queue, ProcessLogger.get_logger().level],
      max_tasks=100
      ) as pool:

      timeout += timeout
      while len(timeout_tasks) > 0:

        results: dict[Future[Result], int] =\
            {pool.schedule(classify_point, args=(abstract_classifier, point_id+1, points[point_id][0], points[point_id][1], params, dataset_props, all_labels), timeout=timeout):point_id
                for point_id in timeout_tasks}

        timeout_tasks = []

        for result in as_completed(results):
          try:
            classification_result = result.result()
            classification_results.append(classification_result.classification)
            robustness_results.append(classification_result.robustness)
            stability_results.append(classification_result.stability)
            stable_count += classification_result.stability_count
            robust_count += classification_result.robustness_count
            classified_points += 1
            pbar.set_postfix_str('ROB={}%, STAB={}%'.format(
              round(sum(robust_count) / (classified_points * len(k_values)) * 100, 1),
              round(sum(stable_count) / (classified_points * len(k_values)) * 100, 1)
            ))
            pbar.update(1)
          except StopIteration:
            break
          except TimeoutError as error:
            print(F"Classification of point {results[result]} took longer than {error.args[1]} seconds")
            timeout_tasks.append(results[result])
          except ProcessExpired as error:
            print("%s. Exit code: %d" % (error, error.exitcode))
            timeout_tasks.append(results[result])
          except Exception as error:
            print("function raised %s" % error)
            print(error.traceback)  # Python's traceback of remote process
            timeout_tasks.append(results[result])

        timeout += timeout

  elapsed_clock: float = time.time() - clock_st

  overall_results: list[list[str]] = [['Overall stability percentage'] +\
      [f'{(stable_points/tot_points)*100}' for stable_points in stable_count]]

  overall_results.append(['Overall robustness percentage'] +\
      [f'{(robust_points/tot_points)*100}' for robust_points in robust_count])

  overall_results.append(['']*len(k_values))
  overall_results.append(['runtime (clock time)', f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_clock))}']+\
                          ['']*(len(k_values)-1))

  results_dir = join(params['base_dirs']['result'], args.config_filepath)
  with open(join(results_dir, 'classification.csv'), 'w', newline='', encoding='utf-8') as classification_file,\
       open(join(results_dir, 'robustness.csv'), 'w', newline='', encoding='utf-8') as robustness_file,\
       open(join(results_dir, 'stability.csv'), 'w', newline='', encoding='utf-8') as stability_file,\
       open(join(results_dir, 'overall_result.csv'), 'w', newline='', encoding='utf-8') as overall_result_file:

    classification_writer = csv.writer(classification_file)
    classification_header = ['Test point #', 'Actual Classification'] + [f' Classification at K={k}' for k in k_values]
    classification_writer.writerow(classification_header)

    robustness_writer = csv.writer(robustness_file)
    robustness_header = ['Test point #'] + [f'Robustness at K={k}' for k in k_values]
    robustness_writer.writerow(robustness_header)

    stability_writer = csv.writer(stability_file)
    stability_header = ['Test point #'] + [f'Stability at K={k}' for k in k_values]
    stability_writer.writerow(stability_header)

    overall_result_writer = csv.writer(overall_result_file)
    overall_result_header = ['At'] + [f'K={k}' for k in k_values]
    overall_result_writer.writerow(overall_result_header)

    classification_writer.writerows(sorted(classification_results, key = lambda x: int(x[0]))) # type: ignore
    robustness_writer.writerows(sorted(robustness_results, key = lambda x: int(x[0]))) # type: ignore
    stability_writer.writerows(sorted(stability_results, key = lambda x: int(x[0]))) # type: ignore
    overall_result_writer.writerows(overall_results)

def sequential_main(params: Configuration, partition_size: int = 20,
         random_state: Optional[int] = None, all_labels: bool = True) -> None:
  global abstract_classifier
  k_values = params['knn_params']['k_values']

  stable_count = zeros(len(k_values))
  robust_count = zeros(len(k_values))

  loader = DataLoader(params['dataset'])

  training_set, test_set, dataset_props = loader.load_datasets(params['base_dirs']['dataset'])

  abstract_classifier.fit(training_set, partition_size, random_state)

  progress_bar = tqdm(
      zip(test_set.points, test_set.labels),
      total=test_set.num_points,
      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_inv_fmt} {postfix}]',
      desc='Verifying'
  )

  epsilon = params['abstraction']['epsilon']

  classified_points = 0
  classification_results: list[list[str]] = []
  robustness_results: list[list[str]] = []
  stability_results: list[list[str]] = []

  clock_st: float = time.time()
  process_st: float = time.process_time()

  for (test_point, test_label) in progress_bar:\


    logger.info("-- Classifying point %s %s with label %s --\n", classified_points + 1 , test_point, test_label)

    perturbation = Perturbation(test_point, epsilon, dataset_props)

    labels = abstract_classifier.get_classification(perturbation, k_values, all_labels)

    classification_result: list[str]  = [str(classified_points+1), str(test_label)]
    robustness_result: list[str]      = [str(classified_points+1)]
    stability_result: list[str]       = [str(classified_points+1)]

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
    classification_results.append(classification_result)
    robustness_results.append(robustness_result)
    stability_results.append(stability_result)
    classified_points += 1
    progress_bar.set_postfix_str('ROB={}%, STAB={}%'.format(
          round(sum(robust_count) / (classified_points * len(k_values)) * 100, 1),
          round(sum(stable_count) / (classified_points * len(k_values)) * 100, 1)
    ))

  elapsed_clock: float = time.time() - clock_st
  elapsed_process: float = time.process_time() - process_st

  overall_results: list[list[str]] = [['Overall stability percentage'] +\
      [f'{(stable_points/classified_points)*100}' for stable_points in stable_count]]

  overall_results.append(['Overall robustness percentage'] +\
      [f'{(robust_points/classified_points)*100}' for robust_points in robust_count])

  overall_results.append(['']*len(k_values))
  overall_results.append(['runtime (clock time)', f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_clock))}']+\
                          ['']*(len(k_values)-1))
  overall_results.append(['runtime (process time)', f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_process))}']+\
                          ['']*(len(k_values)-1))

  results_dir = join(params['base_dirs']['result'], args.config_filepath)
  with open(join(results_dir, 'classification.csv'), 'w', newline='', encoding='utf-8') as classification_file,\
       open(join(results_dir, 'robustness.csv'), 'w', newline='', encoding='utf-8') as robustness_file,\
       open(join(results_dir, 'stability.csv'), 'w', newline='', encoding='utf-8') as stability_file,\
       open(join(results_dir, 'overall_result.csv'), 'w', newline='', encoding='utf-8') as overall_result_file:

    classification_writer = csv.writer(classification_file)
    classification_header = ['Test point #', 'Actual Classification'] + [f' Classification at K={k}' for k in k_values]
    classification_writer.writerow(classification_header)

    robustness_writer = csv.writer(robustness_file)
    robustness_header = ['Test point #'] + [f'Robustness at K={k}' for k in k_values]
    robustness_writer.writerow(robustness_header)

    stability_writer = csv.writer(stability_file)
    stability_header = ['Test point #'] + [f'Stability at K={k}' for k in k_values]
    stability_writer.writerow(stability_header)

    overall_result_writer = csv.writer(overall_result_file)
    overall_result_header = ['At'] + [f'K={k}' for k in k_values]
    overall_result_writer.writerow(overall_result_header)

    classification_writer.writerows(classification_results)
    robustness_writer.writerows(robustness_results)
    stability_writer.writerows(stability_results)
    overall_result_writer.writerows(overall_results)

abstract_classifier: AbstractClassifier = None

if __name__ == "__main__":
  try:

    abstract_classifier: AbstractClassifier = AbstractClassifier()

    args = get_args_parser().parse_args()

    params = Configuration()

    if not exists(params['base_dirs']['logs']):
      mkdir(params['base_dirs']['logs'])

    log_dir: str = join(params['base_dirs']['logs'], args.config_filepath)

    if not exists(log_dir):
      mkdir(log_dir)

    input_file_path = join(params['base_dirs']['config'], f'{args.config_filepath}.toml')
    if not exists(input_file_path):
      raise ValueError(f"Input file {input_file_path} not found:")

    params.load_configuration(input_file_path)

    result_dir = join(params['base_dirs']['result'], args.config_filepath)
    if not exists(params['base_dirs']['result']):

      mkdir(params['base_dirs']['result'])
      mkdir(result_dir)

    elif not exists(result_dir):
      mkdir(result_dir)

    if not args.no_parallel:
      process_logger = ProcessLogger.create_process_logger(
        join(log_dir, 'logs.log'),
        args.log_level.upper())
      process_logger.start()
      parallel_main(params, args.log_level.upper(), args.partition_size, args.random_state, args.all_labels)

      logger.info("Finished classifying !!")

      process_logger.stop()
      process_logger.join()
      process_logger.close()
    else:
      logging.basicConfig(filename=join(log_dir, 'logs.log'),
                        filemode="w", level=args.log_level.upper(), format="%(message)s")

      sequential_main(params, args.partition_size, args.random_state, args.all_labels)

  except Exception:
    logger.exception('An error occurred: ')



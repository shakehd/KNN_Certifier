from __future__ import annotations

import logging
from multiprocessing import Process, Manager
from queue import Queue
from typing import Any, ClassVar, Self
from logging import LogRecord, handlers
from os.path import join

from src.abstract.abstract_classifier import AbstractClassifier

class ProcessLogger(Process):
  _logger: ClassVar[ProcessLogger | None] = None

  def __init__(self: Self, log_filepath: str, level: str) -> None:
    super().__init__()
    self.log_filepath: str =  log_filepath
    self.level: str = level
    self.queue: Queue[LogRecord| None] =  Manager().Queue(-1)
  @classmethod
  def create_process_logger(cls: type[ProcessLogger], filepath: str,
                         level: str) -> ProcessLogger:

    if cls._logger is None:
      cls._logger = ProcessLogger(filepath, level)

    return cls._logger

  @classmethod
  def get_logger(cls: type[ProcessLogger]) -> ProcessLogger:

    if cls._logger is None:
      raise Exception("Process logger is not initialized !!")

    return cls._logger

  @classmethod
  def configure_logger_for_process(cls: type[ProcessLogger],
                                   queue: Queue[LogRecord | None],
                                   level: str) -> None:
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> LogRecord:

      record: LogRecord = old_factory(*args, **kwargs)
      record.point_number = AbstractClassifier.point_number
      return record

    logging.setLogRecordFactory(record_factory)

    h = handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(h)

  def _configure_logger(self) -> None:
    root = logging.getLogger()
    h = logging.FileHandler(self.log_filepath, 'w')
    f = logging.Formatter('Point %(point_number)s\t %(message)s')
    h.setFormatter(f)
    root.setLevel(self.level)
    root.addHandler(h)

  def stop(self):
    self.queue.put_nowait(None)

  def run(self: Self) -> None:
    self._configure_logger()
    while True:
      try:
          record = self.queue.get()
          if record is None:
              break
          logger = logging.getLogger(record.name)
          logger.handle(record)
      except Exception:
          import sys, traceback
          print('Error while logging:', file=sys.stderr)
          traceback.print_exc(file=sys.stderr)

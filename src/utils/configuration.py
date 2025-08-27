from collections import ChainMap
from dataclasses import InitVar, dataclass, field
from typing import Any, Self

import tomli as toml


@dataclass
class Configuration:
  settings: ChainMap[str, Any] = field(init=False)
  init_config_file_path: InitVar[str] = field(default="./settings.toml")

  def __post_init__(self: Self, config_file_path: str) -> None:
    with open(config_file_path, mode="rb") as fp:
      self.settings = ChainMap(toml.load(fp))

  def load_configuration(self: Self, config_file_path: str) -> None:
    with open(config_file_path, mode="rb") as fp:
      self.settings = self.settings.new_child(toml.load(fp))

  def __getitem__(self: Self, key: str) -> Any:
    return self.settings[key]

  def __setitem__(self: Self, key: str, value: Any) -> None:
    self.settings[key] = value

  def __contains__(self: Self, key: str) -> bool:
    return key in self.settings

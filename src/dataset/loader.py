
from dataclasses import dataclass
from typing import Self
from os.path import join
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.utils.base_types import  ArrayNxM
from .file_reader import csv_reader, libsvm_reader
from .dataset import CatFeature, Dataset, DatasetParams, DatasetProps

@dataclass
class DataLoader:
  params: DatasetParams

  def __post_init__(self: Self) -> None:
    self.col_transformer: ColumnTransformer = None # type: ignore

  def load_datasets(self: Self, base_dir: str) -> tuple[Dataset, Dataset, DatasetProps]:

    training_set_path: str = join(base_dir, self.params['training_set'])
    test_set_path: str = join (base_dir, self.params['test_set'])

    if self.params['format'] == 'libsvm':
      training_dataset, training_label = libsvm_reader(training_set_path)
      test_dataset, test_label = libsvm_reader(test_set_path)
    elif self.params['format'] == 'csv':
      training_dataset, training_label = csv_reader(training_set_path)
      test_dataset, test_label = csv_reader(test_set_path)
    else:
      raise ValueError("Invalid dataset format. Only libsvm and csv are supported.")

    scaled_datasets = self.preprocess_data((training_dataset, test_dataset))
    dataset_props = DatasetProps()
    dataset_props.columns = scaled_datasets[0].shape[1]


    if 'category_indexes' in self.params and len(self.params['category_indexes']) > 0:
      one_hot_enc: OneHotEncoder = self.col_transformer.named_transformers_['one_hot_encoder'] # type: ignore
      dataset_props.num_features_start_ix = len(one_hot_enc.get_feature_names_out())# type: ignore

      perturb_features = [] if 'perturb_categories' not in self.params \
                            else self.params['perturb_categories']

      categories_ix = perturb_features + [cat for cat in self.params['category_indexes']\
                                             if cat not in perturb_features]


      idx = 0
      for ix, cat_values in enumerate(one_hot_enc.categories_):
        cat_size = cat_values.size
        dataset_props.cat_features[ix] = CatFeature(
          idx=idx,
          size=cat_size,
          perturb= categories_ix[ix] in perturb_features
        )
        idx += cat_size


    return Dataset(scaled_datasets[0], training_label), \
           Dataset(scaled_datasets[1], test_label), dataset_props


  def preprocess_data(self: Self,
                      dataset: tuple[ArrayNxM, ArrayNxM]) -> tuple[ArrayNxM, ArrayNxM]:

    training_set, test_set = dataset
    if not self.col_transformer:

      transformers: list[tuple[str, TransformerMixin, list[int]]] = []

      categories_ix: list[int] = []  if 'category_indexes' not in self.params \
                                     else self.params['category_indexes']

      perturb_features = [] if 'perturb_categories' not in self.params \
                            else self.params['perturb_categories']

      numerical_features = [] if 'numerical_features' not in self.params \
                            else self.params['numerical_features']

      if perturb_features and not categories_ix:
        raise KeyError('To use "perturb_categories" parameter a non empty list\
                       of categorical feature indexes must be provided.')

      if perturb_features:
        categories_ix = perturb_features + [cat for cat in categories_ix\
                                             if cat not in perturb_features]

      if categories_ix:

        features_categories: list[list[str| int]] = []
        for idx in categories_ix:

          categories: list[str| int] = list(set(training_set[:, idx]).union(set(test_set[:, idx])))

          if idx in numerical_features:
            categories.sort()

          features_categories.append(categories)

        transformers.append(('one_hot_encoder',
                            OneHotEncoder(categories=features_categories,drop='if_binary',
                                          sparse_output=False),
                            categories_ix))

      numerical_ix: list[int] = list(set(range(dataset[0].shape[1])) - set(categories_ix))
      DatasetProps.num_features_ix = numerical_ix

      transformers.append(('num_scaler',
                          MinMaxScaler(feature_range=(0, 1)),
                          numerical_ix))

      self.col_transformer = ColumnTransformer(transformers)
      training_set = self.col_transformer.fit_transform(training_set) # type: ignore
      test_set = self.col_transformer.transform(test_set) # type: ignore

    return  training_set, test_set # type: ignore
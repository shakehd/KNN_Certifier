# Robustness Verification of *k*-Nearest neighbours

Implementation of an abstract interpretation-based tool for proving robustness and stability properties of *k*NN classifiers.

## Requirements
- Python3

## Installation
To install this toool you need to clone or download this repository and run the commands:
```[bash]
pip install -r requirements.txt
```
This will install the following dependencies:
- joblib==1.4.2
- nptyping==2.5.0
- numpy==1.26.4
- optype==0.9.2
- pandas==2.2.3
- Pebble==5.1.1
- python-dateutil==2.9.0.post0
- pytz==2025.1
- scikit-learn==1.6.1
- scipy==1.15.2
- scipy-stubs==1.15.2.1
- six==1.17.0
- threadpoolctl==3.6.0
- tomli==2.2.1
- tqdm==4.67.1
- tzdata==2025.1

## Usage

To run this tool simply launch the following command inside the repo folder:

```[bash]
python certifier.py CONFIGFILE <arguments>
```
where `CONFIGFILE` is a config file present inside the `configs` folder (or the
one configured in the `.settings.toml` configuration file), meanwhile `<arguments>`
can be one of the following:

|  Arg | Description  |
|---|---|
| --random-state RANDOM | Random seed used when partitioning the dataset. |
| --partition-size SIZE | Maximum number of data points in a partition (default 100). |
| --log  {INFO,DEBUG,ERROR}  | Log level used during the verification phase (default ERROR).  |
| --all-labels  | Compute all labels. (default False)  |
| --no-parallel | Classify points sequentially (default False).  |
| -h, --help  | Show help message and exit.  |

For example

```[bash]
python certify.py fourclass_5_percent --partition-size 100 --log info
```
[!WARNING]
For parallizzation the FORK method is used. Since this is only available in Unix systems this tool will not work on windows systems.

## Results
After the certification process is finished the tool will save the results in 4 files:
- **classification.csv**: Contains the classifications results for each value of k.
- **robustness.csv**: Contains the robustness results for each value of k.
- **stability.csv**: Contains the stability results for each value of k.
- **overall_result.csv**: Contains the overall robustness and stability percentage and
                          runtime information.

## Configurations

The tool requires two configuration files to work properly:

- settings.toml: A TOML configuration file specifying the folders containing the datasets
                 and configurations for the verification process of a dataset.

- *verification*.toml: A TOML configuration file specifying the settings needed to verify a
                     a dataset.

### settings.toml

The settings.toml has the following form:
```
[base_dirs]
config = "./configs"
dataset = "./datasets"
result = "./results"
logs = "./logs""
```
where the `base_dirs` contains the following settings:

- `config`: directory where the configuration files are located (default ./config).
- `dataset_dir`: directory where datasets are located  (default ./dataset).
- `result`: directory where the verification results are saved  (default ./result).
- `logs`: directory where the logs are saved  (default ./logs).

### verification.toml

The *verification*.toml has the following form:
```
[knn_params]
k_values = [list of k values]
distance_metric = "distance metric (euclidean or manhattan)"

[dataset]
format = "dataset format (libsvm or csv)"
training_set = "training dataset name"
test_set = "test dataset name"

[abstraction]
epsilon = epsilon value
```
It has three section:

- `knn_params`:
  - `k_values`: list of possible values for the number of nearest neighbours to consider for each prediction.
  - `distance_metric`: The metric used to measure the distance between data points

- `dataset`:
  - `format`: The format of the dataset which can be *libsvm* or *csv*.
  - `training_set`: The name of the file that contains the training data
  - `test_set`:  The name of the file that contains the test data
  - `category_indexes`: The indexes (zero-based) corresponding to categorical features in the input
  - `numerical_features`: The indexes (zero-based) corresponding to numerical features in the input (if missing then all features are considered numerical)
  - `perturb_categories`: The index  (zero-based) of the category to perturb for Noise_Cat pertubation

- `abstraction`:
  - `epsilon`: The perturbation magnitude value.

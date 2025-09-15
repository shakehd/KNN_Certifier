# Robustness Verification of *k*-Nearest neighbours

Implementation of the tool for certifiying the exact robustness and stability properties of *k*NN classifiers described in the paper [[1]](#1).

## Requirements
- Python version 3.12+

## Installation
To install this tool you need to clone or download this repository and run the following command from inside the repository main folder:
```[bash]
python3 -m pip install -r requirements.txt
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

## Folder Structure
There are three main folders:

 - `configs`: contains the configuration files used for the certifications phase. The structure of these files is described below. Each configuration file use the following naming conventions
```
datasetname_epsilon_percent.tml
```
   where `epsilon` are represented as percentages. For example to certify the dataset pendigits with ε = 0.01, the
   corresponding configuration file is pendigits_1_percent.toml.
 - `datasets`: contains the pairs of traning and test datasets used for experimentation results. Each pair is in the _csv_ or _libsvm_ format.
 - `src`: contains the source code of the tool.

## Usage

To run this tool simply launch the following command from inside the main repository folder:

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
| --log  {INFO,DEBUG,ERROR}  | Log level used during the ceritifcation phase (default ERROR).  |
| --retrieve-all-labels  | Consider all labels during classification (default False)  |
| --no-parallel | Classify points sequentially (default False).  |
| --no-early-stopping | Disable early stopping optimization (default False).  |
| --no-majority-prunning | Disable majority prunning optimization (default False).  |
| --no-early-majority-detection | Disable early majority detection optimization (default False).  |
| --no-path-length-bounds | Disable path length bounds optimization (default False).  |
| --consider-all-permutations | Consider all samples permutations during classification (default False).  |
| -h, --help  | Show help message and exit.  |

For example

```[bash]
python certify.py fourclass_1_percent 
```
For the experimental results reported in the paper [[1]](#1) we used WSL2 Arch Linux on a machine with 12 core CPU and 32 GB of RAM and executed the certifier using the default values of each argument.

> [!WARNING]
> For parallizzation the FORK method is used. Since this is only available in Linux systems this tool will not work on windows systems. To use on Windows OS use the --no-parallel flag.

## Results
After the certification process is finished the tool will save the results under the folder _.\results\CONFIGFILE_. This folder contains the following files::
- **classification.csv**: contains the classifications results for each value of k.
- **robustness.csv**: contains the robustness results for each value of k.
- **stability.csv**: contains the stability results for each value of k.
- **overall_result.csv**: contains the runtime information and the overall robustness and stability percentage of the _k_NN for the dataset and ε specified in the _CONFIGFILE_ file.
                          
## Configurations

The tool requires two configuration files to work properly:

- settings.toml: A TOML configuration file specifying the folders containing the datasets
                 and configurations for the verification process of a dataset.

- *dataset_n_percent*.toml: A TOML configuration file specifying the settings needed to certify
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
where the `base_dirs` tag contains the following keys:

- `config`: directory where the configuration files are searched (default ./config).
- `dataset_dir`: directory where datasets are searched  (default ./dataset).
- `result`: directory where the certification results are saved  (default ./result).
- `logs`: directory where the logs are saved  (default ./logs).

### dataset_n_percent.toml

The *dataset_n_percent*.toml has the following schema:
```
[knn_params]
k_values = [list of k values]

[dataset]
format = "..."
training_set = "..."
test_set = "..."
category_indexes = "..."
numerical_features = "..."
perturb_categories = "..."

[abstraction]
epsilon = ...
```
It has three section:

- `knn_params`:
  - `k_values`: list of possible values for the number of nearest neighbours to consider for each prediction.

- `dataset`:
  - `format`: The format of the dataset which can be *libsvm* or *csv*.
  - `training_set`: The name of the file that contains the training data. This file is searched in the folder specified in the settings.toml file.
  - `test_set`:  The name of the file that contains the test data. This file is searched in the folder specified in the settings.toml file.
  - `category_indexes`: The indexes (zero-based) corresponding to the categorical features in the input.
  - `numerical_features`: The indexes (zero-based) corresponding to the numerical features in the input (if missing then all features are considered numerical).
  - `perturb_categories`: The index (zero-based) of the categories to perturb in the Noise_Cat pertubation.

- `abstraction`:
  - `epsilon`: The perturbation magnitude value.

## References
<a id="1">[1]</a>
Francesco Ranzato, Ahmad Shakeel, and Marco Zanella. 2025. Exact Robustness Certification of k-Nearest Neighbors. In Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security (CCS ’25), October 13–17, 2025, Taipei, Taiwan. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3719027.3765140


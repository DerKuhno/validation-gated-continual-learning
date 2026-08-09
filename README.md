# Validation-Gated Continual Learning

A project for researching the viability and effectiveness of applying a Continual Learning approach for the ESA-ADB satellite telemetry dataset. It explores how this machine learning paradigm can be used for anomaly detection in satellite telemetry data from the European Space Agency (ESA).

## Current State
**Note:** This repository is currently being tested. If you encounter any bugs or issues, please open an [issue](https://github.com/DerKuhno/validation-gated-continual-learning/issues).

## Installation

To set up the project environment, you need to have Conda installed. You can create the conda environment with the `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate vgcl
```

## Data

Download the ESA Anomalies Dataset from the link https://doi.org/10.5281/zenodo.12528696 (ESA-Mission1 and ESA-Mission2) into the "data" folder.

```bash
mkdir validation-gated-continual-learning/data
cd validation-gated-continual-learning/data
# Download ESA-Mission1
wget "https://zenodo.org/records/12528696/files/ESA-Mission1.zip" -O ESA-Mission1.zip
# Download ESA-Mission2
wget "https://zenodo.org/records/12528696/files/ESA-Mission2.zip" -O ESA-Mission2.zip
```

```bash
unzip ESA-Mission1.zip -d .
unzip ESA-Mission2.zip -d .
```

## Data preperation

For data preperation, we refer to the datapreperation given in https://github.com/kplabs-pl/ESA-ADB. The relevant part is:

### Github and Environment
```bash
git clone https://github.com/kplabs-pl/ESA-ADB
cd ESA-ADB
conda env create -f environment.yml
conda activate timeeval
```
### Data preperation
```bash
cd ESA-ADB
# Mission1: 
PYTHONPATH=Path/to/ESA-ADB python notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py \
    Path/to/validation-gated-continual-learning/data/ESA-Mission1

# Mission2: 
PYTHONPATH=Path/to/ESA-ADB python notebooks/data-prep/Mission2_semiunsupervised_prep_from_raw.py \
    Path/to/validation-gated-continual-learning/data/ESA-Mission2

# Copying into the data folder
mv Path/to/ESA-ADB/data/preprocessed Path/to/validation-gated-continual-learning/data/
```
Afterwards, the file structure should look as described below.

## Usage

The main entry point of the project is `run.py`. To run the project, execute:

```bash
python run.py
```

Experiments can be run by changing hyperparameters in `experiment_config.py` and `CL_methods/cl_method_params.py` or iterating through them in `run.py`:

```python
hyperparameter_grid = {
    'seed': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    'buffer_size': [70_000, 140_000, 350_000]
}
```

The project also includes Jupyter notebooks for data exploration and experimentation in the `notebooks` directory. You can run them using Jupyter Notebook or JupyterLab.

## Project Structure

```
.
├── .gitignore
├── LICENSE
├── README.md
├── environment.yml
├── run.py
├── reproducibility-test.py
├── data/
│   ├── ESA-Mission1/
│   │   ├── anomaly_types.csv
│   │   ├── channels.csv
│   │   ├── labels.csv
│   │   ├── telecommands.csv
│   │   ├── channels/
│   │   └── telecommands/
│   ├── ESA-Mission2/
│   │   └── (same structure as ESA-Mission1)
│   └── preprocessed/
│       ├── datasets.csv
│       ├── esa-anomaly-dataset-metadata.json
│       └── multivariate/
├── results/
│   └── <timestamp>/
│       ├── anomaly_prediction.csv
│       ├── errors.csv
│       ├── experiment.log
│       ├── hyper_params.json
│       ├── metrics.csv
│       └── reconstruction.csv
├── scripts/
│   └── threshold_param_search.py
└── src/
    ├── Main_loop.py
    ├── base_model.py
    ├── dataset.py
    ├── dataset_manager.py
    ├── detector.py
    ├── esaadb_dataset.py
    ├── experiment_config.py
    ├── general_utils.py
    ├── telemanom_model.py
    ├── CL_methods/
    │   ├── EpisodeBalancedBuffer.py
    │   ├── agem.py
    │   ├── cl_method_params.py
    │   ├── continual_learning_manager.py
    │   ├── der.py
    │   ├── er.py
    │   ├── finetuning.py
    │   ├── gem.py
    │   ├── gss_greedy.py
    │   ├── latent_greedy.py
    │   ├── loss_greedy.py
    │   ├── lower_bound.py
    │   ├── mir.py
    │   ├── ser.py
    │   ├── upper_bound.py
    │   ├── utils.py
    │   └── variance_greedy.py
    └── metrics/
        ├── ESA_ADB_metrics.py
        ├── MetricsEvaluator.py
        ├── metric.py
        └── utils/
            ├── convert_to_events.py
            ├── datasets.py
            └── encode_params.py
```

## Reproducing the Experiments.
It's highly recommended to split up the hyperparameter grid and run multiple experiments at the same time, if possible.

The experiments from Table 1 / Figure 4 can be done with the following dictionaries in run.py:

### Static
```python
hyperparameter_grid = {
    'continual_learning': [False],
    'seed': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
}
```
### Fine-tuning, ER, DER++, SER, GEM, A-GEM, MIR
```python
hyperparameter_grid = {
    'continual_learning': [True],
    'cl_method': ["finetuning", "er", "der" "ser", "gem", "agem", "mir"],
    'seed': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
}
```
### Lower Bound
```python
hyperparameter_grid = {
    'continual_learning': [True],
    'cl_method': ["lower_bound"],
    'seed': [42, 43, 44],
}
```
### Upper Bound
```python
hyperparameter_grid = {
    'continual_learning': [True],
    'cl_method': ["upper_bound"],
    'seed': [42],
}
```

The experiments from Figure 6 can be done with the following dictionary in run.py:

```python
hyperparameter_grid = {
    'continual_learning': [True],
    'cl_method': ["er", "der", "ser", "mir", "gem", "agem"],
    'seed': [42, 43, 44],
    'replay_strat': ['episode_balanced'],
    'buffer_size': [1_400, 3_500, 7_000, 14_000, 35_000, 70_000, 140_000, 350_000],
}
```

The experiments from Figure 7 can be done with the following dictionary in run.py:

```python
hyperparameter_grid = {
    'continual_learning': [True],
    'cl_method': ["er"],
    'seed': [42, 43, 44],
    'replay_strat': ['episode_balanced', 'loss_greedy', 'gss_greedy', 'latent_greedy', 'variance_greedy'],
    'buffer_size': [1_400, 70_000],
}
```

The experiments from Figure 8 can be done with the following dictionary in run.py:

### Fine-tuning, ER, SER
```python
hyperparameter_grid = {
    'collection': ["ESA-Mission2"],
    'dataset': ["21_months"],
    'continual_learning': [True],
    'cl_method': ["finetuning", "er", "ser"],
    'seed': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
}
```

The experiments from Figure 8 can be done with the following dictionary in run.py:

### Fine-tuning, ER, SER
```python
hyperparameter_grid = {
    'collection': ["ESA-Mission2"],
    'dataset': ["21_months"],
    'continual_learning': [True],
    'cl_method': ["finetuning", "er", "ser"],
    'seed': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
}
```
### Static
```python
hyperparameter_grid = {
    'collection': ["ESA-Mission2"],
    'dataset': ["21_months"],
    'continual_learning': [False],
    'seed': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
}
```

The experiments from Figure A.9 can be done with the following dictionary in run.py:

```python
hyperparameter_grid = {
    'continual_learning': [True],
    'fixed_epochs': [True],
    'fixed_num_epochs': [1, 5, 10],
    'cl_method': ["finetuning", "er", "ser"],
    'seed': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
}
```

## Citation

```bibtex
@article{KUHN2026971,
title = {Validation-gated continual learning for anomaly detection in satellite telemetry},
journal = {Acta Astronautica},
volume = {249},
pages = {971-985},
year = {2026},
issn = {0094-5765},
doi = {https://doi.org/10.1016/j.actaastro.2026.07.065},
url = {https://www.sciencedirect.com/science/article/pii/S0094576526005187},
author = {Nils Kuhn and Bruno {Sánchez Gómez} and Natalia {Moreno Blasco} and Polona Caserman and Federico Antonello}
}
```

## Authors and Acknowledgment

- Bruno Sanchez Gomez
- Nils Kuhn
- Natalia Moreno Blasco

Interns in OPS-GAA at ESOC (ESA).

We want to acknowledge the work European Space Agency Benchmark for Anomaly Detection in
Satellite Telemetry by Kotowski et al.

K. Kotowski, C. Haskamp, J. Andrzejewski, B. Ruszczak, J. Nalepa,
D. Lakey, P. Collins, A. Kolmas, M. Bartesaghi, J. Martinez-Heras, G. D.
Canio, European space agency benchmark for anomaly detection in satellite telemetry, arXiv preprint arXiv:2406.17826 (2024). arXiv:2406.17826,
doi:https://doi.org/10.48550/arXiv.2406.17826.


We also want to acknowledge the Avalanche CL library which we used as basis for some of the CL methods.
A. Carta, L. Pellegrini, A. Cossu, H. Hemati, V. Lomonaco, Avalanche: A
pytorch library for deep continual learning, Journal of Machine Learning
Research 24 (363) (2023) 1–6.
URL http://jmlr.org/papers/v24/23-0130.html

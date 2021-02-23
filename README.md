# Benchmarking of ambulance forecasting models.

This repo contains the data, code and results summary to support the publication 'Forecasting the daily demand for emergency medical ambulances in England and Wales: A benchmark model and external validation'.

The purpose of the analysis contained in this repository is to 'model selection and generation of benchmark accuracy for forecasting the number of calls that result in the dispatch of one or more ambulances.'

# General description of the repo

The analysis is broken into model selection (2 stages) and benchmarking.  Stage 1 model selection screens 15 methods on a single time series (Trust level).  Stage 2 model selection compares Facebook Prophet and an ensemble of Prophet and Regression with ARIMA errors on 7 unseen time series + the trust.  Benchmarking runs the selected model - the ensemble - on an unseen test set from the 7 series.

Results are recorded for a horizon of 7 to 84 days spaced at 7 day intervals for:
* Mean absolute percentage error (MASE) 
* Symmetric Mean Absolute Percentage Error (sMAPE)
* Root Mean Squared Error (RMSE)
* 80% Prediction interval coverage
* 95% Prediction interval coverage
* 60, 70, 80, 90, 95% intervals for the final benchmark model.

(stage 1 also contains a 365 day forecast)

These results are recorded via time series cross validation.  These can be found under `./results` Each time series and measure has its own .csv file.  The columns in the results file represents a forecast horizon (e.g. 7, 14, 21, ..., 84).  The rows represent a cross validation fold. 

# Repo Structure

```
swast_benchmarking
├── LICENSE
├── analysis
│   ├── benchmark
│   |   ├── 00_ensemble-tscv-parallel.ipynb
│   ├── model_selection
│   |   ├── stage1
│   │   |   ├── 00_batch_run_stage1.ipynb
│   │   |   ├── [15 X analysis notebooks; 1 X R script]
│   |   ├── stage2
│   │   |   ├── 00_batch_run_stage2.ipynb
│   │   |   ├── [2 X analysis notebooks]
├── binder
│   ├── environment.yml
├── data
│   ├── [2 x.csv; inc. 1 x R formatted time series.]
├── paper
│   ├── appendix
│   ├── figures
│   ├── tables
├── results
│   ├── benchmark
│   |   ├── [.csv]
│   ├── model_selection
│   |   ├── stage1
│   │   |   ├── [.csv]
│   |   ├── stage1
│   │   |   ├── [.csv]
│   ├── benchmark_summary.ipynb
│   ├── summary_stage_1.ipynb
│   ├── summary_stage_2.ipynb
├── README.md
└── results_summary_main.ipynb
```

# Steps to reproduce the results reported in the paper.

Use the environment `ambo_benchmark` detailed in `binder/environment.yml`.  

## Installation of conda environment:

The analysis is written in a mix of standard Python (3.x) and modern data science libraries. It is recommended that users first install 'Anaconda'. Anaconda includes 'conda' (a package manager).

Anaconda: https://www.anaconda.com/download/

To install the correct version of the libraries it is recommended that the provided conda environment is used. To create and activate a bootcomp environment:

* Windows -> Open Anaconda prompt. Mac/linux -> Open a terminal
* Navigate to the swast-benchmarking directory
* Run the following command: 

```    
conda env create -f binder/environment.yml
```
This will fetch and install the libraries in a conda environment `ambo_benchmark`

To activate the enviroment run the following command: 

```
conda activate ambo_benchmark
```

More help on environments can be found here: https://conda.io/docs/user-guide/tasks/manage-environments.html

## Quickly recreate the tables, charts and figures in the paper:

The results files are included in the repo.  To quickly reproduce the tables, figures and statistics included in the results section of the paper run the top level notebook `results_summary_main.ipynb`.  The notebook will display 'most' of the outputs; the appendix tables are not practical to output.  All outputs including the appendix can also be found in appendix/

## How to conduct a full reproduction run.

> Note: the below executes a large number of Time Series Cross Validation procedures.  Depending on your machine it may take several hours to run.

Run the following notebooks (the order does not matter)

* ./analysis/model_selection/stage1/00_batch_run_stage1.ipynb
* ./analysis/model_selection/stage2/00_batch_run_stage2.ipynb
* ./analysis/benchmark/00_ensemble-tscv-parallel.ipynb
* ./results_summary_main.ipynb




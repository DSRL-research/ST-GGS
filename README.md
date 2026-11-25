# Spatio-Temporal GraphSAGE for Generalizable Power Grid Representations

Code for the training and evaluation of ST-GGS, a heterogeneous GraphSAGE model that
learns a token for each grid element from its history, static attributes and network
neighbourhood. Every experiment runs on the ACTIVSg2000 and ACTIVSg10k grids.

## Setup

```
pip install -r requirements.txt
```

Commands run from this folder. Place the data as follows, or pass `--data-root`:

```
datasets/
  ACTIVSg/                 yearly load and renewable CSV files of both grids
  ACTIVSg2000/             ACTIVSg2000.m, case_ACTIVSg2000.m, ACTIVSg2000.aux
  ACTIVSg10k/              case_ACTIVSg10k.m, ACTIVSg10k.aux
```

The grids and their time series are available from the TAMU Electric Grid Test Case
Repository.

## Power-flow labels

The state-estimation task needs an AC power-flow solution for every hour.

```
python -m stggs.power_flow.generate_labels --dataset activsg2000
python -m stggs.power_flow.generate_labels --dataset activsg10k
```

## Training

`--variant` is `st_ggs` or `st_ggs_kcl`, which adds the Kirchhoff term to the loss.
All settings live in [stggs/config.py](stggs/config.py).

```
python -m stggs.training.train_st_ggs --dataset activsg2000 --variant st_ggs
python -m stggs.training.train_st_ggs --dataset activsg2000 --variant st_ggs_kcl
python -m stggs.training.train_st_ggs --dataset activsg10k --variant st_ggs
python -m stggs.training.train_st_ggs --dataset activsg10k --variant st_ggs_kcl
```

The Transformer baselines are one model per horizon and element type:

```
python -m stggs.training.train_specialists --dataset activsg2000
python -m stggs.training.train_specialists --dataset activsg10k
```

Checkpoints are written to `runs/<dataset>/<variant>/` and `runs/<dataset>/specialists/`.

## Evaluation

Run each command for both grids and both variants.

| Result | Command |
|---|---|
| Table 1, forecasting | `python -m stggs.evaluation.forecasting --dataset activsg2000 --variant st_ggs` |
| Table 2, direct estimation | `python -m stggs.evaluation.state_estimation --dataset activsg2000 --variant st_ggs` |
| Table 2, impute-solve | `python -m stggs.evaluation.impute_solve --dataset activsg2000 --variant st_ggs` |
| Table 3, zero-shot transfer | `python -m stggs.evaluation.transfer --source activsg2000 --target activsg10k --variant st_ggs` |

Then collect the results into tables under `runs/tables/`:

```
python -m stggs.evaluation.tables
```

## Layout

```
stggs/
  config.py        hyperparameters of every experiment
  data/            ACTIVSg loading, windows, graph construction, observability
  models/          ST-GGS, temporal encoder, heads, physics, Transformer baseline
  power_flow/      pandapower network, AC solver, label generation
  training/        ST-GGS and Transformer specialist training
  evaluation/      one module per reported result, plus table generation
```

## Citation

Toderean, L., Cioara, T., Michalakopoulos, V., Sarmas, E., and Anghel, I. Spatio-Temporal
GraphSAGE Framework for Generalizable Power Grid Representations. NeurIPS 2026 Workshop on
AI Foundations for Power Grids: From Models to Deployment at Scale.

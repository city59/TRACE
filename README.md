# TRACE

Code for **TRACE: Tracing Candidate-Specific Evidence Paths for Multi-Behavior Sequential Recommendation**.

A GRU first encodes the multi-behavior history into contextual states. For every candidate, TRACE scores ordered historical paths with a unified path utility that combines candidate relevance (N), transition compatibility (E) and endpoint support (F). A forward dynamic program (Algorithm 1) retrieves the MAP path and its top-Ω alternatives, and a suffix dynamic program over the full path space gives the connection probabilities η, which bias attention in a structure-guided path transformer. Finally, a learned gate fuses the scores of the MAP and alternative path representations.

| Paper section | Code (`trace.py`) |
|---|---|
| 4.2 Interaction Encoding | `_encode_history`, `_candidate_representation` |
| 4.3 Path Utility and Selection | `_transition_utility`, `_candidate_utilities`, `_ranked_paths` |
| 4.4 Structure-Guided Path Transformer | `_log_suffix_weights`, `PathTransformerBlock`, `_score` |
| 4.5 Recommendation and Training | `_score`, `calculate_loss` |

The code is built on [RecBole](https://github.com/RUCAIBox/RecBole) and [MBHT](https://github.com/yuh-yang/MBHT-KDD22). Only the RecBole modules needed to run TRACE are kept.

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.9 and PyTorch 1.12 (CUDA).

## Datasets

The benchmarks are built from the raw interaction logs, which keep every timestamp:

| Paper name   | Raw file | Source | Behaviors |
|--------------|----------|--------|-----------|
| Taobao       | `UserBehavior.csv` (or its `.zip`) | [Tianchi 649](https://tianchi.aliyun.com/dataset/649) | purchase, cart, favorite, view |
| Retailrocket | `events.csv` | [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) | purchase, cart, view |
| IJCAI        | `data_format1.zip` (`user_log_format1.csv`) | [Tianchi 47](https://tianchi.aliyun.com/dataset/47) | purchase, cart, favorite, view |

```bash
python scripts/preprocess.py --source taobao --input path/to/UserBehavior.csv.zip
```

```bash
python scripts/preprocess.py --source retailrocket --input path/to/events.csv
```

```bash
python scripts/preprocess.py --source ijcai --input path/to/data_format1.zip
```

Each command writes `dataset/<name>/<name>.{train,valid,test}.inter` and `<name>.stats.json`. The protocol follows Sec. 3 and Sec. 5.1:

- Each user's interactions are sorted by timestamp. Interactions with the same timestamp keep the order of the raw log. Timestamps are stored in hours (IJCAI only records the day). They give θ in Eqs. (2) and (3).
- Chronological leave-one-out: each user's last purchase is the test target and the second-to-last purchase is the validation target. Every earlier purchase is a training target. The history of a target is everything the user did before it.
- Each validation and test target is ranked against 100 distinct negatives, sampled by item popularity in the training portion. The negatives are stored in the files, so every model is evaluated on the same candidates.

Filtering (Sec. 5.1): items need at least 5 interactions and users at least 2 purchases, applied repeatedly until both hold. Targets without any preceding interaction are dropped, and each history keeps its most recent 200 interactions. These settings and the sampling seed are fixed in the script. Behavior token `0` is the target behavior (purchase).

## Run

```bash
python run_TRACE.py --dataset retailrocket --gpu_id 0
```

```bash
python run_TRACE.py --dataset taobao --gpu_id 0
```

```bash
python run_TRACE.py --dataset ijcai --gpu_id 0
```

`--dataset` accepts only these three datasets. Logs, checkpoints and a result JSON are written to `run_artifacts/trace/`. Each run reports HR@10 and NDCG@10 on the test targets, using the checkpoint with the best validation NDCG@10. The random seed is fixed (2026).

## Hyperparameters

All settings are fixed; the command line only selects the dataset, the GPU and memory-only batch sizes. Settings from the paper (Sec. 5.1):

| Parameter | Value |
|---|---|
| Optimizer | AdamW, learning rate 1e-3, weight decay 1e-4 |
| Embedding size d | 64 |
| Batch size | 2048 |
| Initialization | Xavier |

Per-dataset values (Sec. 5.4), set in `run_TRACE.py`:

| Dataset | Path temperature τ | Transformer blocks G | Non-MAP paths Ω |
|---|---|---|---|
| Retailrocket | 0.4 | 2 | 2 |
| Taobao | 0.4 | 2 | 6 |
| IJCAI | 0.6 | 3 | 8 |

## Code structure

| File | Content |
|---|---|
| `scripts/preprocess.py` | builds the three benchmarks from the raw logs |
| `run_TRACE.py` | training and evaluation entry point |
| `recbole/model/sequential_recommender/trace.py` | TRACE model (Eqs. 1–14, Algorithm 1) |
| `recbole/properties/model/TRACE.yaml` | model settings |
| `recbole/` (other files) | RecBole data loading, training loop and HR/NDCG evaluation |

## Implementation notes

The paper does not fix these settings. The defaults below are implementation choices:

- 10 attention heads per block. Since 64 is not divisible by 10, each head has ceil(64/10) = 7 dimensions (70 in total), the output projection maps back to 64, and scores are scaled by sqrt(d/H) as in Eq. 10. Feed-forward width is 256 and dropout is 0.1.
- 1 uniformly sampled training negative per positive.
- Early stopping on validation NDCG@10 with patience 10, at most 200 epochs.
- Each batch of 2048 is processed in gradient-checkpointed micro-batches (`--micro_batch`) to fit in GPU memory. This is still one optimizer step per batch.

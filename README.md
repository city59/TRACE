# TRACE

Code for **Target Reasoning via Adaptive Combinatorial Evidence Optimization for Recommendation** (TRACE).

For every candidate item, TRACE scores ordered historical paths by event relevance (N), transition compatibility (E) and endpoint support (F). Exact dynamic programming retrieves the MAP path and the top-Ω alternatives, and a suffix DP gives the next-link probabilities η. A structure-guided path transformer then encodes each path. Finally, a learned gate fuses the MAP score and the utility-weighted alternative score.

The code is built on [RecBole](https://github.com/RUCAIBox/RecBole) and the [MBHT](https://github.com/yuh-yang/MBHT-KDD22) data pipeline.

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.9 and PyTorch 1.12 (CUDA).

## Datasets

Download the MBHT datasets ([Google Drive](https://drive.google.com/file/d/1OFT_5Xp_az-GSHIl7QEPB9zhulbooLzE/view?usp=sharing)) and unzip them into `./dataset/`:

| Paper name   | Folder       | Behaviors                      |
|--------------|--------------|--------------------------------|
| Taobao       | `tmall_beh`  | purchase, cart, favorite, view |
| Retailrocket | `retail_beh` | purchase, cart, view           |
| IJCAI        | `ijcai_beh`  | purchase, cart, favorite, view |

Behavior token `0` is the target behavior (purchase). If a dataset has an `item_time_list` field, its timestamps are used for the temporal costs. Otherwise the historical position is used as the time unit.

## Run

```bash
python run_TRACE.py --dataset retail_beh --gpu_id 0
```

```bash
python run_TRACE.py --dataset tmall_beh --gpu_id 0
```

```bash
python run_TRACE.py --dataset ijcai_beh --gpu_id 0
```

Logs, checkpoints and a result JSON are written to `run_artifacts/trace/`. Each run reports HR@10 and NDCG@10. Every positive is ranked against 100 popularity-based negatives, following the MBHT protocol.

The paper repeats each experiment with 10 seeds:

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do python run_TRACE.py --dataset retail_beh --seed $s; done
```

## Hyperparameters

Settings from the paper (Sec. 5.1), used as defaults:

| Parameter | Value |
|---|---|
| Optimizer | AdamW, learning rate 1e-3 |
| Embedding size d | 64 |
| Batch size | 2048 |
| Initialization | Xavier |
| L2 coefficient λ_reg | 1e-4 (added to the loss, Eq. 16) |
| Path temperature τ | searched in {0.1, 0.2, 0.4, 0.6, 0.8, 1.0} |
| Transformer blocks J | searched in {1, 2, 3, 4, 5} |
| Max non-MAP paths Ω | searched in {1, 2, 4, 6, 8, 10} |

Best values per dataset (Sec. 5.4), applied automatically:

| Dataset | τ | J | Ω |
|---|---|---|---|
| Retailrocket | 0.4 | 2 | 2 |
| Taobao | 0.4 | 2 | 6 |
| IJCAI | 0.6 | 3 | 8 |

Override them with `--tau`, `--blocks` and `--omega`. Other options are listed in `python run_TRACE.py -h` and `recbole/properties/model/TRACE.yaml`.

## Ablations (Sec. 5.3)

Use `--ablation <name>`:

| Name | Variant |
|---|---|
| `wo_N` / `wo_E` / `wo_F` | set N, E or F to zero |
| `map_only` | Ω = 0 |
| `soft_only` | remove the MAP branch |
| `wo_bias` | remove log R from attention |
| `wo_cq` | remove z_c from the attention query |
| `wo_tf` | remove the path transformer (mean of path inputs) |
| `wo_gru` | cumulative sums of interaction embeddings instead of the GRU |
| `shared_A` | one projection A shared across behaviors |

## Code structure

| File | Content |
|---|---|
| `recbole/model/sequential_recommender/trace.py` | TRACE model (Eqs. 1–16) |
| `recbole/properties/model/TRACE.yaml` | default hyperparameters |
| `run_TRACE.py` | training / evaluation entry point |
| `tests/test_trace.py` | checks the DP against brute-force path enumeration |

Run the tests with:

```bash
python -m unittest tests.test_trace
```

## Implementation notes

The paper does not fix these settings. The defaults below are implementation choices:

- 10 attention heads per block. Since 64 is not divisible by 10, each head has ceil(64/10) = 7 dimensions (70 in total), the output projection maps back to 64, and scores are scaled by sqrt(d/H) as in Eq. 12. Feed-forward width is 256 and dropout is 0.1.
- 1 uniformly sampled training negative per positive (`--train_neg_num`).
- The MBHT files only provide train/test. 10% of the training rows are held out for validation and early stopping (patience 10, at most 200 epochs).
- Each batch of 2048 is processed in gradient-checkpointed micro-batches (`--micro_batch`) to fit in GPU memory. This is still one optimizer step per batch.

The original MBHT README is kept in `README_MBHT.md`. The earlier `MBHT` and `C3Comb` models are left unchanged.

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

The three datasets (preprocessed as in MBHT) are included in this repository. Retailrocket is stored as plain files. Taobao and IJCAI exceed GitHub's file size limit, so they are stored as xz-compressed archives in `dataset/archives/`. Unpack them once before running:

```bash
python scripts/prepare_datasets.py
```

The script extracts the archives into `./dataset/` and verifies every file against `dataset/archives/SHA256SUMS`.

| Paper name   | Folder       | Behaviors                      | Stored as |
|--------------|--------------|--------------------------------|-----------|
| Taobao       | `tmall_beh`  | purchase, cart, favorite, view | `tmall_beh.tar.xz` |
| Retailrocket | `retail_beh` | purchase, cart, view           | plain files |
| IJCAI        | `ijcai_beh`  | purchase, cart, favorite, view | `ijcai_beh.tar.xz.part00`, `.part01` |

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

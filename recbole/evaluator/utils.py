import itertools

import numpy as np
import torch


def pad_sequence(sequences, len_list, pad_to=None, padding_value=0):
    max_len = np.max(len_list) if pad_to is None else pad_to
    min_len = np.min(len_list)
    device = sequences[0].device
    if max_len == min_len:
        result = torch.cat(sequences, dim=0).view(-1, max_len)
    else:
        extra_len_list = np.subtract(max_len, len_list).tolist()
        padding_nums = max_len * len(len_list) - np.sum(len_list)
        padding_tensor = torch.tensor([-np.inf], device=device).repeat(padding_nums)
        padding_list = torch.split(padding_tensor, extra_len_list)
        result = list(itertools.chain.from_iterable(zip(sequences, padding_list)))
        result = torch.cat(result)

    return result.view(-1, max_len)


def trunc(scores, method):
    try:
        cut_method = getattr(np, method)
    except NotImplementedError:
        raise NotImplementedError("module 'numpy' has no function named '{}'".format(method))
    scores = cut_method(scores)
    return scores


def cutoff(scores, threshold):
    return np.where(scores > threshold, 1, 0)


def _binary_clf_curve(trues, preds):
    trues = (trues == 1)

    desc_idxs = np.argsort(preds)[::-1]
    preds = preds[desc_idxs]
    trues = trues[desc_idxs]

    unique_val_idxs = np.where(np.diff(preds))[0]
    threshold_idxs = np.r_[unique_val_idxs, trues.size - 1]

    tps = np.cumsum(trues)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps

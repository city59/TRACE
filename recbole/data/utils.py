# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/9, 2020/9/17, 2020/8/31, 2021/2/20, 2021/3/1
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li, Haoran Cheng, Jiawei Guan
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com, chenghaoran29@foxmail.com, guanjw@ruc.edu.cn

"""
recbole.data.utils
########################
"""

from recbole.data.dataloader import *
from recbole.sampler import Sampler, RepeatableSampler
from recbole.utils import ModelType


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    if config['MODEL_TYPE'] != ModelType.SEQUENTIAL:
        raise ValueError('Only the sequential TRACE model is supported.')
    from .dataset import SequentialDataset
    return SequentialDataset(config)


def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase == 'train':
        return TrainDataLoader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    phases = ['train', 'test']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']
    sampler = None
    train_sampler, valid_sampler, test_sampler = None, None, None

    if train_neg_sample_args['strategy'] != 'none':
        if not config['repeatable']:
            sampler = Sampler(phases, built_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_sampler = sampler.set_phase('train')

    if eval_neg_sample_args['strategy'] != 'none':
        if sampler is None:
            config['repeatable'] = False
            if not config['repeatable']:
                sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        # valid_sampler = sampler.set_phase('valid')
        test_sampler = sampler.set_phase('test')

    return train_sampler, test_sampler

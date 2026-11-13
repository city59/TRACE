from recbole.data.dataloader import *
from recbole.sampler import Sampler, RepeatableSampler
from recbole.utils import ModelType


def create_dataset(config):
    if config['MODEL_TYPE'] != ModelType.SEQUENTIAL:
        raise ValueError('Only the sequential TRACE model is supported.')
    from .dataset import SequentialDataset
    return SequentialDataset(config)


def get_dataloader(config, phase):
    if phase == 'train':
        return TrainDataLoader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader


def create_samplers(config, dataset, built_datasets):
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
        test_sampler = sampler.set_phase('test')

    return train_sampler, test_sampler

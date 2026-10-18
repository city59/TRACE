import argparse
import json
from logging import getLogger
import os
from pathlib import Path
import re
import time
from datetime import datetime

import torch
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader, create_samplers
from recbole.model.sequential_recommender.mbht import MBHT
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color


def validate_c3comb_benchmark_manifest(data_path, dataset_name):
    """Reject timestamped benchmarks generated with the superseded protocol.

    C3Comb's paper-aligned protocol uses every eligible target occurrence and
    the complete observable prefix.  Failing here is safer than silently
    training the revised model on an older, capped benchmark artifact.
    """
    manifest_path = Path(data_path) / f'{dataset_name}.manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'C3Comb requires a benchmark manifest at {manifest_path}. '
            'Regenerate the timestamped benchmark with '
            'scripts/build_timestamp_benchmark.py.'
        )

    with manifest_path.open('r', encoding='utf-8') as manifest_file:
        manifest = json.load(manifest_file)

    violations = []
    history_policy = manifest.get('history_policy')
    max_history = manifest.get('max_history')
    if history_policy != 'complete observable prefix' or max_history not in (None, 0):
        violations.append(
            'history must use the complete observable prefix '
            f'(found history_policy={history_policy!r}, max_history={max_history!r})'
        )

    target_policy = manifest.get('target_supervision_policy')
    if target_policy != 'every eligible target-behavior occurrence':
        violations.append(
            'every eligible target-behavior occurrence must be supervised '
            f'(found target_supervision_policy={target_policy!r})'
        )
    if manifest.get('max_train_targets_per_user') not in (None, 0):
        violations.append(
            'per-user target caps are not allowed '
            f"(found max_train_targets_per_user={manifest['max_train_targets_per_user']!r})"
        )

    if violations:
        details = '; '.join(violations)
        raise ValueError(
            f'Benchmark {manifest_path} follows the superseded protocol: {details}. '
            'Regenerate it with the current scripts/build_timestamp_benchmark.py '
            '(default --max-history 0).'
        )
    return manifest


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MBHT', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='tmall_beh', help='Benchmarks for session-based rec.')
    parser.add_argument(
        '--validation',
        action='store_true',
        help=(
            'Split the official training file into train/validation for model '
            'selection, then evaluate the best checkpoint once on the official test file.'
        ),
    )
    parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=0, help='Override the dataset-specific training batch size when positive.')
    parser.add_argument('--eval_batch_size', type=int, default=0, help='Override the dataset-specific evaluation batch size when positive.')
    parser.add_argument(
        '--eval_mode', type=str, default='pop100',
        help=(
            'RecBole dataset evaluation protocol: full, uniN, or popN. '
            'The model no longer fixes the candidate-set size.'
        ),
    )
    parser.add_argument(
        '--variant',
        choices=['baseline', 'c3route', 'c3route_gt', 'c3comb'],
        default='baseline',
    )
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument(
        '--learner', choices=['adam', 'adamw'], default='adam'
    )
    parser.add_argument(
        '--lr_scheduler', choices=['none', 'warmup_cosine'], default='none'
    )
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument(
        '--eval_step', type=int, default=1,
        help='Run validation after every N training epochs (default: 1).',
    )
    parser.add_argument(
        '--stopping_step', type=int, default=30,
        help='Stop after this many consecutive validations without improvement (default: 30).',
    )
    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--clip_grad_norm', type=float, default=5.0)
    parser.add_argument(
        '--max_seq_length', type=int, default=200,
        help=(
            'Maximum length for RecBole-generated legacy sequences. '
            'Timestamped C3Comb uses the prefix stored by the benchmark and '
            'does not crop it inside the model.'
        ),
    )
    parser.add_argument('--route_topk', type=int, default=6)
    parser.add_argument('--route_weight', type=float, default=3.0)
    parser.add_argument('--route_aux_weight', type=float, default=0.25)
    parser.add_argument('--route_coverage_weight', type=float, default=0.01)
    parser.add_argument('--route_negatives', type=int, default=4)
    parser.add_argument('--route_mode', choices=['local', 'global'], default='global')
    parser.add_argument('--route_stages', type=int, default=3)
    parser.add_argument('--route_temperature', type=float, default=0.7)
    parser.add_argument('--route_rerank_topr', type=int, default=500)
    parser.add_argument('--horizon_bins', type=str, default='0.25,1,6,24,168')
    parser.add_argument('--combo_hidden_size', type=int, default=32)
    parser.add_argument(
        '--target_behavior_token', type=str, default='0',
        help='Dataset token denoting the supervised target behavior.',
    )
    parser.add_argument('--combo_negatives', type=int, default=32)
    parser.add_argument('--combo_uniform_weight', type=float, default=0.4)
    parser.add_argument('--combo_pop_weight', type=float, default=0.3)
    parser.add_argument('--combo_hard_weight', type=float, default=0.3)
    parser.add_argument('--combo_pop_alpha', type=float, default=0.75)
    parser.add_argument(
        '--combo_linear_kernel', type=int, choices=[0, 1], default=1,
        help='Use exact linear-time separable global-path inference.',
    )
    parser.add_argument('--combo_kernel_channels', type=int, default=8)
    parser.add_argument(
        '--combo_closed_form_kernel', type=int, choices=[0, 1], default=1,
        help='Use the vectorized exact rank-one positive transition kernel.',
    )
    parser.add_argument(
        '--combo_length_bucket', type=int, choices=[0, 1], default=1,
        help='Group similar complete-prefix lengths within shuffled batches.',
    )
    parser.add_argument('--combo_temperature', type=float, default=0.7)
    parser.add_argument('--combo_min_temperature', type=float, default=0.05)
    parser.add_argument('--combo_logit_scale', type=float, default=8.0)
    parser.add_argument('--combo_learnable_scale', type=int, choices=[0, 1], default=1)
    parser.add_argument('--combo_candidate_chunk_size', type=int, default=64)
    parser.add_argument(
        '--report_repeat_groups', type=int, choices=[0, 1], default=1
    )
    parser.add_argument(
        '--run_tag', type=str, default='',
        help='Optional filesystem-safe suffix that preserves earlier results.'
    )
    parser.add_argument(
        '--artifact_root', type=str, default='run_artifacts',
        help='Writable root for logs, checkpoints, and result JSON files.',
    )
    parser.add_argument(
        '--timestamped', action='store_true',
        help='Use explicit train/valid/development/confirmation benchmark files.'
    )
    parser.add_argument(
        '--evaluation_split', choices=['development', 'confirmation'],
        default='development',
        help='Confirmation is used only after the method is frozen.'
    )
    parser.add_argument('--resume', type=str, default='', help='Resume a full trainer checkpoint.')
    parser.add_argument(
        '--initialize_from', type=str, default='',
        help=(
            'Warm-start matching model parameters from a checkpoint without '
            'restoring the optimizer or early-stopping state.'
        ),
    )
    parser.add_argument(
        '--freeze_warmstarted', action='store_true',
        help='When warm-starting, optimize only parameters absent from the source checkpoint.',
    )
    parser.add_argument('--resume_log', type=str, default='', help='Log whose completed validation epochs should be retained.')
    parser.add_argument('--prior_elapsed_seconds', type=float, default=0.0, help='Wall-clock time already spent before resuming.')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = get_args()

    # configurations initialization
    config_dict = {
        'USER_ID_FIELD': 'session_id',
        'load_col': None,
        # 'neg_sampling': {'uniform':1},
        'neg_sampling': None,
        'benchmark_filename': (
            ['train', 'valid', 'development', 'confirmation']
            if args.timestamped else ['train', 'test']
        ),
        'alias_of_item_id': ['item_id_list'],
        'topk': [5, 10, 101],
        'metrics': ['Recall', 'NDCG', 'MRR'],
        'valid_metric': (
            'balanced_ndcg@10'
            if args.variant == 'c3comb' and args.report_repeat_groups
            else 'NDCG@10'
        ),
        'eval_args':{
            'mode':args.eval_mode,
            'order':'TO'
            },
        'gpu_id':args.gpu_id,
        'show_progress':False,
        'seed':args.seed,
        'reproducibility':True,
        'epochs':args.epochs,
        'learning_rate':args.learning_rate,
        'learner': args.learner,
        'lr_scheduler': args.lr_scheduler,
        'warmup_steps': args.warmup_steps,
        'weight_decay':args.weight_decay,
        'clip_grad_norm': {'max_norm': args.clip_grad_norm},
        'eval_step': args.eval_step,
        'stopping_step':args.stopping_step,
        'min_delta':args.min_delta,
        "MAX_ITEM_LIST_LENGTH":args.max_seq_length,
        "train_batch_size": args.batch_size if args.batch_size > 0 else (8 if args.timestamped else (32 if args.dataset == "ijcai_beh" else 64)),
        "eval_batch_size": args.eval_batch_size if args.eval_batch_size > 0 else (8 if args.timestamped else (24 if args.dataset == "ijcai_beh" else 128)),
        "hyper_len":10 if args.dataset == "ijcai_beh" else 6,
        "scales":[10, 4, 20],
        "enable_hg":1,
        "enable_ms":1,
        "customized_eval":0 if (args.variant == 'c3comb' or args.timestamped) else 1,
        "enable_c3route": int(args.variant in ('c3route', 'c3route_gt')),
        "enable_temporal": int(args.variant == 'c3route_gt'),
        "route_topk": args.route_topk,
        "route_weight": args.route_weight,
        "route_aux_weight": args.route_aux_weight,
        "route_coverage_weight": args.route_coverage_weight,
        "route_negatives": args.route_negatives,
        "route_mode": args.route_mode,
        "route_stages": args.route_stages,
        "route_temperature": args.route_temperature,
        "route_rerank_topr": args.route_rerank_topr,
        "horizon_bins": [float(value) for value in args.horizon_bins.split(',')],
        "hidden_size": args.combo_hidden_size if args.variant == 'c3comb' else 64,
        "target_behavior_token": args.target_behavior_token,
        "combo_negatives": args.combo_negatives,
        "combo_uniform_weight": args.combo_uniform_weight,
        "combo_pop_weight": args.combo_pop_weight,
        "combo_hard_weight": args.combo_hard_weight,
        "combo_pop_alpha": args.combo_pop_alpha,
        "combo_linear_kernel": args.combo_linear_kernel,
        "combo_kernel_channels": args.combo_kernel_channels,
        "combo_closed_form_kernel": args.combo_closed_form_kernel,
        "combo_length_bucket": args.combo_length_bucket,
        "combo_temperature": args.combo_temperature,
        "combo_min_temperature": args.combo_min_temperature,
        "combo_logit_scale": args.combo_logit_scale,
        "combo_learnable_scale": args.combo_learnable_scale,
        "combo_candidate_chunk_size": args.combo_candidate_chunk_size,
        "report_repeat_groups": bool(args.report_repeat_groups),
        "checkpoint_dir": os.path.join(args.artifact_root, 'saved'),
        "abaltion":""
    }

    if args.dataset == "retail_beh":
        config_dict['scales'] = [5, 4, 20]
        config_dict['hyper_len'] = 6
        
    model_name = 'C3Comb' if args.variant == 'c3comb' else 'MBHT'
    args.model = model_name
    config = Config(model=model_name, dataset=f'{args.dataset}', config_dict=config_dict)
    # config['device']="cpu"
    init_seed(config['seed'], config['reproducibility'])

    benchmark_manifest = None
    if args.variant == 'c3comb' and args.timestamped:
        benchmark_manifest = validate_c3comb_benchmark_manifest(
            config['data_path'], args.dataset
        )

    # logger initialization
    init_logger(config, log_root=os.path.join(args.artifact_root, 'log'))
    logger = getLogger()

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)
    logger.info(config)

    if benchmark_manifest is not None:
        logger.info(
            'validated paper-aligned C3Comb benchmark manifest: %s',
            benchmark_manifest,
        )

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    built_datasets = dataset.build()
    if args.timestamped:
        train_dataset, valid_dataset, development_dataset, confirmation_dataset = built_datasets
        train_sampler, valid_sampler = create_samplers(
            config, dataset, [train_dataset, valid_dataset]
        )
        _, development_sampler = create_samplers(
            config, dataset, [train_dataset, development_dataset]
        )
        _, confirmation_sampler = create_samplers(
            config, dataset, [train_dataset, confirmation_dataset]
        )
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
        valid_data = get_dataloader(config, 'test')(config, valid_dataset, valid_sampler, shuffle=False)
        evaluation_dataset = development_dataset if args.evaluation_split == 'development' else confirmation_dataset
        evaluation_sampler = development_sampler if args.evaluation_split == 'development' else confirmation_sampler
        official_test_data = get_dataloader(config, 'test')(
            config, evaluation_dataset, evaluation_sampler, shuffle=False
        )
    else:
        train_dataset, test_dataset = built_datasets
        train_sampler, test_sampler = create_samplers(
            config, dataset, built_datasets
        )
    if not args.timestamped and args.validation:
        train_dataset.shuffle()
        fit_dataset, valid_dataset = train_dataset.split_by_ratio([1 - args.valid_portion, args.valid_portion])
        train_data = get_dataloader(config, 'train')(config, fit_dataset, None, shuffle=True)
        valid_data = get_dataloader(config, 'test')(config, valid_dataset, None, shuffle=False)
        official_test_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)
    elif not args.timestamped:
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
        valid_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)
        official_test_data = valid_data

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    initialization_report = None
    if args.initialize_from:
        initialization_checkpoint = torch.load(args.initialize_from, map_location='cpu')
        source_state = initialization_checkpoint.get(
            'state_dict', initialization_checkpoint
        )
        incompatible = model.load_state_dict(source_state, strict=False)
        initialization_report = {
            'checkpoint': args.initialize_from,
            'missing_keys': sorted(incompatible.missing_keys),
            'unexpected_keys': sorted(incompatible.unexpected_keys),
        }
        logger.info(
            'warm-started matching parameters from %s; missing=%d unexpected=%d',
            args.initialize_from,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        if args.freeze_warmstarted:
            trainable_names = set(incompatible.missing_keys)
            for name, parameter in model.named_parameters():
                parameter.requires_grad = name in trainable_names
            logger.info(
                'froze warm-started parameters; trainable tensors=%d parameters=%d',
                sum(parameter.requires_grad for parameter in model.parameters()),
                sum(
                    parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            )
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    if args.resume:
        trainer.resume_checkpoint(args.resume)

    # model training and evaluation
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    validation_history = []
    if args.resume_log:
        validation_pattern = re.compile(
            r'epoch\s+(-?\d+)\s+evaluating\s+\[.*?valid_score:\s*([0-9.]+)'
        )
        with open(args.resume_log, 'r', encoding='utf-8') as resume_log_file:
            for line in resume_log_file:
                match = validation_pattern.search(line)
                if match and int(match.group(1)) >= 0:
                    validation_history.append({
                        'epoch': int(match.group(1)) + 1,
                        'valid_score': float(match.group(2)),
                    })

    def record_validation(epoch_idx, valid_score):
        validation_history.append({
            'epoch': int(epoch_idx + 1),
            'valid_score': float(valid_score),
            'optimizer_step': int(trainer.global_train_step),
            'learning_rate': float(trainer.optimizer.param_groups[0]['lr']),
        })

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config['show_progress'],
        callback_fn=record_validation,
    )
    training_sampling_stats = (
        model.sampling_statistics() if args.variant == 'c3comb' else None
    )
    checkpoint = torch.load(trainer.saved_model_file, map_location='cpu')
    best_epoch_zero_based = int(checkpoint.get('epoch', -1))
    best_epoch_one_based = (
        best_epoch_zero_based + 1 if best_epoch_zero_based >= 0 else None
    )
    test_result = trainer.evaluate(
        official_test_data,
        load_best_model=True,
        show_progress=config['show_progress'],
    )
    selected_checkpoint_sampling_stats = (
        model.sampling_statistics() if args.variant == 'c3comb' else None
    )
    elapsed_seconds = args.prior_elapsed_seconds + time.perf_counter() - run_started
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else 0.0
    )

    logger.info(set_color('best validation result', 'yellow') + f': {best_valid_result}')
    evaluation_label = (
        f'{args.evaluation_split} result' if args.timestamped else 'official test result'
    )
    logger.info(set_color(evaluation_label, 'yellow') + f': {test_result}')

    result_dir = os.path.join(args.artifact_root, 'results')
    os.makedirs(result_dir, exist_ok=True)
    split_name = (
        f'timestamped-{args.evaluation_split}'
        if args.timestamped else ('early-stop' if args.validation else 'test-as-valid')
    )
    safe_run_tag = re.sub(r'[^A-Za-z0-9_.-]+', '-', args.run_tag).strip('-')
    run_suffix = f'-{safe_run_tag}' if safe_run_tag else ''
    result_path = os.path.join(
        result_dir,
        f"{args.dataset}-{args.variant}-{split_name}-seed{args.seed}-epochs{args.epochs}{run_suffix}.json"
    )
    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'dataset': args.dataset,
        'variant': args.variant,
        'seed': args.seed,
        'epochs': args.epochs,
        'epochs_completed': (
            validation_history[-1]['epoch'] if validation_history else 0
        ),
        'eval_step': args.eval_step,
        'stopping_step': args.stopping_step,
        'min_delta': args.min_delta,
        'learning_rate': args.learning_rate,
        'learner': args.learner,
        'lr_scheduler': args.lr_scheduler,
        'warmup_steps': args.warmup_steps,
        'optimizer_steps_completed': int(trainer.global_train_step),
        'best_optimizer_step': int(
            checkpoint.get('global_train_step', 0)
        ),
        'final_learning_rate': float(
            trainer.optimizer.param_groups[0]['lr']
        ),
        'weight_decay': args.weight_decay,
        'clip_grad_norm': args.clip_grad_norm,
        'model_class': model_name,
        'max_seq_length': args.max_seq_length,
        'legacy_training_split_enabled': bool(args.validation),
        'timestamped_split_files': bool(args.timestamped),
        'valid_portion': args.valid_portion,
        'train_batch_size': config['train_batch_size'],
        'eval_batch_size': config['eval_batch_size'],
        'best_epoch_zero_based': best_epoch_zero_based,
        'best_epoch_one_based': best_epoch_one_based,
        'best_checkpoint_stage': (
            'initialization' if best_epoch_zero_based < 0 else 'trained_epoch'
        ),
        'best_valid_score': float(best_valid_score),
        'best_valid_result': dict(best_valid_result),
        'validation_selection_metric': config['valid_metric'],
        'validation_history': validation_history,
        'checkpoint': trainer.saved_model_file,
        'resumed_from': args.resume or None,
        'initialized_from': initialization_report,
        'freeze_warmstarted': bool(args.freeze_warmstarted),
        'elapsed_seconds': elapsed_seconds,
        'peak_gpu_memory_mb': peak_gpu_memory_mb,
        'route_topk': args.route_topk,
        'route_weight': args.route_weight,
        'route_aux_weight': args.route_aux_weight,
        'route_coverage_weight': args.route_coverage_weight,
        'route_negatives': args.route_negatives,
        'route_mode': args.route_mode,
        'route_stages': args.route_stages,
        'route_temperature': args.route_temperature,
        'route_rerank_topr': args.route_rerank_topr,
        'horizon_bins': [float(value) for value in args.horizon_bins.split(',')],
        'combo_hidden_size': args.combo_hidden_size,
        'combo_negatives': args.combo_negatives,
        'combo_uniform_weight': args.combo_uniform_weight,
        'combo_pop_weight': args.combo_pop_weight,
        'combo_hard_weight': args.combo_hard_weight,
        'combo_pop_alpha': args.combo_pop_alpha,
        'combo_linear_kernel': bool(args.combo_linear_kernel),
        'combo_kernel_channels': args.combo_kernel_channels,
        'combo_closed_form_kernel': bool(args.combo_closed_form_kernel),
        'combo_length_bucket': bool(args.combo_length_bucket),
        'combo_temperature': args.combo_temperature,
        'combo_min_temperature': args.combo_min_temperature,
        'combo_logit_scale': args.combo_logit_scale,
        'combo_learnable_scale': bool(args.combo_learnable_scale),
        'combo_candidate_chunk_size': args.combo_candidate_chunk_size,
        'report_repeat_groups': bool(args.report_repeat_groups),
        'training_sampling_statistics': training_sampling_stats,
        'selected_checkpoint_sampling_statistics': (
            selected_checkpoint_sampling_stats
        ),
        'run_tag': safe_run_tag or None,
        'timestamped': bool(args.timestamped),
        'evaluation_mode': args.eval_mode,
        'target_behavior_token': args.target_behavior_token,
        'benchmark_manifest': benchmark_manifest,
        'evaluation_split': args.evaluation_split if args.timestamped else 'legacy-test',
        'evaluation_result': dict(test_result),
    }
    if args.variant == 'c3comb':
        for legacy_key in (
            'route_topk', 'route_weight', 'route_aux_weight',
            'route_coverage_weight', 'route_negatives', 'route_mode',
            'route_stages', 'route_temperature', 'route_rerank_topr',
            'horizon_bins',
        ):
            payload.pop(legacy_key, None)
    else:
        for combo_key in (
            'combo_hidden_size', 'combo_negatives', 'combo_uniform_weight',
            'combo_pop_weight', 'combo_hard_weight', 'combo_pop_alpha',
            'combo_linear_kernel', 'combo_kernel_channels',
            'combo_closed_form_kernel',
            'combo_length_bucket',
            'combo_temperature', 'combo_min_temperature',
            'combo_logit_scale', 'combo_learnable_scale',
            'combo_candidate_chunk_size',
            'report_repeat_groups', 'training_sampling_statistics',
            'selected_checkpoint_sampling_statistics',
        ):
            payload.pop(combo_key, None)
    with open(result_path, 'w', encoding='utf-8') as result_file:
        json.dump(payload, result_file, indent=2, sort_keys=True)
    logger.info(f"result artifact: {result_path}")

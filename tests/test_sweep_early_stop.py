import numpy as np

from pufferlib.sweep import Random, ParetoGenetic


def _minimal_sweep_config():
    return {
        'method': 'Random',
        'metric': 'score',
        'goal': 'maximize',
        'metric_distribution': 'linear',
        'downsample': 1,
        'use_gpu': False,
        'prune_pareto': True,
        'early_stop_quantile': 0.3,
        'train': {
            'learning_rate': {
                'distribution': 'log_normal',
                'min': 1e-4,
                'max': 1e-2,
                'scale': 'auto',
            },
        },
    }


def test_random_early_stop_detects_nan_in_nested_loss():
    sweep = Random(_minimal_sweep_config())
    logs = {'loss': {'policy': np.nan, 'value': 0.1, 'total': 0.2}}
    assert sweep.early_stop(logs, 'score') is True
    assert logs['is_loss_nan'] is True


def test_pareto_genetic_early_stop_detects_nan_in_nested_loss():
    sweep = ParetoGenetic(_minimal_sweep_config())
    logs = {'loss': {'entropy': 0.01, 'total': float('nan')}}
    assert sweep.early_stop(logs, 'score') is True
    assert logs['is_loss_nan'] is True


def test_random_early_stop_ignores_finite_losses():
    sweep = Random(_minimal_sweep_config())
    logs = {'loss': {'policy': 0.1, 'value': 0.2, 'total': 0.3}}
    assert sweep.early_stop(logs, 'score') is False
    assert 'is_loss_nan' not in logs


def test_random_observe_stores_normalized_params():
    sweep = Random(_minimal_sweep_config())
    hypers = {'train': {'learning_rate': 1e-3}}
    sweep.observe(hypers, score=1.0, cost=10.0)
    stored = sweep.success_observations[0]['input']
    assert isinstance(stored, np.ndarray)
    assert stored.shape == (1,)

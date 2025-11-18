"""
Módulo de modelos de Machine Learning.
"""

from .trainer import (
    create_baseline,
    create_models,
    train_model,
    train_all_models,
    create_model_pipelines,
    compare_models,
    select_best_model,
    save_model,
)

from .comparison import (
    compare_approaches,
    create_comparison_table,
    plot_approach_comparison,
)

__all__ = [
    'create_baseline',
    'create_models',
    'train_model',
    'train_all_models',
    'create_model_pipelines',
    'compare_models',
    'select_best_model',
    'save_model',
    'compare_approaches',
    'create_comparison_table',
    'plot_approach_comparison',
]

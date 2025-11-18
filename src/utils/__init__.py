"""
Módulo de utilitários para o projeto de ML.
"""

from .model_evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    print_classification_report,
    compare_models,
    plot_model_comparison,
    plot_feature_importance,
    generate_evaluation_report,
)

__all__ = [
    'evaluate_model',
    'plot_confusion_matrix',
    'print_classification_report',
    'compare_models',
    'plot_model_comparison',
    'plot_feature_importance',
    'generate_evaluation_report',
]


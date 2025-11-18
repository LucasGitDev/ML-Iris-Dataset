"""
Módulo de comparação de abordagens de modelagem.

Este módulo fornece funções para comparar diferentes estratégias
de pré-processamento e modelagem.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compare_approaches(
    models_without_scaling: Dict[str, Any],
    models_with_scaling: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test_scaled: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compara diferentes abordagens de pré-processamento/modelagem.
    
    Abordagem 1: Sem normalização (DecisionTree, NaiveBayes)
    Abordagem 2: Com normalização (k-NN, MLP, Perceptron)
    
    Parameters
    ----------
    models_without_scaling : dict
        Modelos que não usam normalização
    models_with_scaling : dict
        Modelos que usam normalização
    X_test : array-like
        Features de teste (não normalizadas)
    y_test : array-like
        Target de teste
    X_test_scaled : array-like, optional
        Features de teste normalizadas
        
    Returns
    -------
    pd.DataFrame
        DataFrame comparando abordagens
    """
    if X_test_scaled is None:
        X_test_scaled = X_test
    
    results = []
    
    # Avaliar modelos sem normalização
    for model_name, model in models_without_scaling.items():
        y_pred = model.predict(X_test)
        results.append({
            'Abordagem': 'Sem Normalização',
            'Modelo': model_name,
            'Acurácia': accuracy_score(y_test, y_pred),
            'Precisão': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='macro', zero_division=0)
        })
    
    # Avaliar modelos com normalização
    for model_name, model in models_with_scaling.items():
        # Verificar se é pipeline ou modelo simples
        if hasattr(model, 'predict') and hasattr(model, 'steps'):
            # É um pipeline
            y_pred = model.predict(X_test)
        else:
            # É modelo simples que precisa dados normalizados
            y_pred = model.predict(X_test_scaled)
        
        results.append({
            'Abordagem': 'Com Normalização',
            'Modelo': model_name,
            'Acurácia': accuracy_score(y_test, y_pred),
            'Precisão': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='macro', zero_division=0)
        })
    
    df_results = pd.DataFrame(results)
    
    return df_results


def create_comparison_table(
    comparison_df: pd.DataFrame,
    metric: str = 'Acurácia'
) -> pd.DataFrame:
    """
    Cria tabela comparativa de abordagens agrupada por abordagem.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame com resultados de compare_approaches
    metric : str
        Métrica principal para ordenação
        
    Returns
    -------
    pd.DataFrame
        Tabela agrupada por abordagem
    """
    summary = comparison_df.groupby('Abordagem').agg({
        'Acurácia': ['mean', 'max', 'min'],
        'Precisão': 'mean',
        'Recall': 'mean',
        'F1-Score': 'mean'
    }).round(4)
    
    summary.columns = [
        'Acurácia (média)', 'Acurácia (máx)', 'Acurácia (mín)',
        'Precisão (média)', 'Recall (média)', 'F1-Score (média)'
    ]
    
    return summary


def plot_approach_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'Acurácia',
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Cria visualização comparativa de abordagens.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame com resultados de compare_approaches
    metric : str
        Métrica para visualizar
    figsize : tuple
        Tamanho da figura
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Gráfico 1: Comparação por modelo
    approaches = comparison_df['Abordagem'].unique()
    models = comparison_df['Modelo'].unique()
    
    x = np.arange(len(models))
    width = 0.35
    
    for i, approach in enumerate(approaches):
        approach_data = comparison_df[comparison_df['Abordagem'] == approach]
        values = [approach_data[approach_data['Modelo'] == model][metric].values[0] 
                 if len(approach_data[approach_data['Modelo'] == model]) > 0 else 0
                 for model in models]
        ax1.bar(x + i*width, values, width, label=approach, alpha=0.8)
    
    ax1.set_xlabel('Modelo', fontsize=12)
    ax1.set_ylabel(metric, fontsize=12)
    ax1.set_title(f'Comparação de {metric} por Modelo e Abordagem', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Média por abordagem
    summary = comparison_df.groupby('Abordagem')[metric].mean()
    bars = ax2.bar(summary.index, summary.values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    
    # Adicionar valores nas barras
    for bar, value in zip(bars, summary.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Abordagem', fontsize=12)
    ax2.set_ylabel(f'{metric} (média)', fontsize=12)
    ax2.set_title(f'Média de {metric} por Abordagem', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


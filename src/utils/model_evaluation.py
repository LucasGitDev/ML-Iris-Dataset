"""
Módulo de avaliação de modelos de Machine Learning.

Este módulo fornece funções para avaliar o desempenho de modelos de classificação,
incluindo métricas, visualizações e relatórios detalhados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    model_name: str = "Modelo",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Avalia um modelo de classificação e retorna métricas principais.
    
    Parameters
    ----------
    model : sklearn classifier
        Modelo treinado para avaliação
    X_test : array-like
        Features de teste
    y_test : array-like
        Labels verdadeiros de teste
    y_pred : array-like, optional
        Predições do modelo. Se None, será calculado usando model.predict(X_test)
    model_name : str
        Nome do modelo para exibição
    verbose : bool
        Se True, imprime as métricas
        
    Returns
    -------
    dict
        Dicionário com métricas de avaliação
    """
    if y_pred is None:
        y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    
    if verbose:
        print("=" * 60)
        print(f"AVALIAÇÃO DO MODELO: {model_name}")
        print("=" * 60)
        print(f"\nMÉTRICAS PRINCIPAIS:")
        print(f"  Acurácia:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precisão (macro):   {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro):   {metrics['f1_macro']:.4f}")
        print(f"\n  Precisão (weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (weighted):   {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (weighted): {metrics['f1_weighted']:.4f}")
        print("=" * 60)
    
    return metrics


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    model_name: str = "Modelo",
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = False
) -> None:
    """
    Plota a matriz de confusão de forma visual e informativa.
    
    Parameters
    ----------
    y_test : array-like
        Labels verdadeiros
    y_pred : array-like
        Labels preditos
    class_names : list, optional
        Nomes das classes. Se None, será usado np.unique(y_test)
    model_name : str
        Nome do modelo para o título
    figsize : tuple
        Tamanho da figura
    normalize : bool
        Se True, normaliza a matriz (mostra percentuais)
    """
    if class_names is None:
        class_names = sorted(np.unique(y_test))
    
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title_suffix = " (Normalizada)"
    else:
        fmt = 'd'
        title_suffix = ""
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Frequência' if not normalize else 'Proporção'}
    )
    
    plt.title(f'Matriz de Confusão - {model_name}{title_suffix}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Classe Verdadeira', fontsize=12)
    plt.xlabel('Classe Predita', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Estatísticas adicionais
    print("\n" + "=" * 60)
    print("ANÁLISE DA MATRIZ DE CONFUSÃO")
    print("=" * 60)
    
    total = cm.sum() if not normalize else len(y_test)
    correct = np.trace(cm) if not normalize else np.trace(cm) * len(y_test) / len(class_names)
    
    print(f"\nTotal de amostras: {len(y_test)}")
    print(f"Predições corretas: {int(correct)} ({correct/len(y_test)*100:.2f}%)")
    print(f"Predições incorretas: {len(y_test) - int(correct)} ({(len(y_test) - correct)/len(y_test)*100:.2f}%)")
    
    # Análise por classe
    print("\nAnálise por classe:")
    for i, class_name in enumerate(class_names):
        if normalize:
            correct_class = cm[i, i] * len(y_test) / len(class_names)
            total_class = len(y_test) / len(class_names)
        else:
            correct_class = cm[i, i]
            total_class = cm[i, :].sum()
        
        accuracy_class = (correct_class / total_class * 100) if total_class > 0 else 0
        print(f"  {class_name}: {int(correct_class)}/{int(total_class)} corretas ({accuracy_class:.2f}%)")


def print_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    model_name: str = "Modelo"
) -> None:
    """
    Imprime relatório de classificação formatado e detalhado.
    
    Parameters
    ----------
    y_test : array-like
        Labels verdadeiros
    y_pred : array-like
        Labels preditos
    class_names : list, optional
        Nomes das classes
    model_name : str
        Nome do modelo
    """
    print("=" * 60)
    print(f"RELATÓRIO DE CLASSIFICAÇÃO - {model_name}")
    print("=" * 60)
    print("\n" + classification_report(y_test, y_pred, target_names=class_names))
    print("=" * 60)


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_preds: Optional[Dict[str, np.ndarray]] = None
) -> pd.DataFrame:
    """
    Compara múltiplos modelos e retorna um DataFrame com as métricas.
    
    Parameters
    ----------
    models : dict
        Dicionário com nome do modelo como chave e modelo treinado como valor
    X_test : array-like
        Features de teste
    y_test : array-like
        Labels verdadeiros
    y_preds : dict, optional
        Dicionário com predições pré-calculadas. Se None, será calculado
        
    Returns
    -------
    pd.DataFrame
        DataFrame com métricas comparativas
    """
    if y_preds is None:
        y_preds = {name: model.predict(X_test) for name, model in models.items()}
    
    results = []
    
    for name in models.keys():
        y_pred = y_preds[name]
        metrics = {
            'Modelo': name,
            'Acurácia': accuracy_score(y_test, y_pred),
            'Precisão (macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall (macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1-Score (macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
        }
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Acurácia', ascending=False)
    
    print("=" * 80)
    print("COMPARAÇÃO DE MODELOS")
    print("=" * 80)
    print("\n" + df_results.to_string(index=False))
    print("\n" + "=" * 80)
    print(f"Melhor modelo: {df_results.iloc[0]['Modelo']} "
          f"(Acurácia: {df_results.iloc[0]['Acurácia']:.4f})")
    print("=" * 80)
    
    return df_results


def plot_model_comparison(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_preds: Optional[Dict[str, np.ndarray]] = None,
    metric: str = 'Acurácia',
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Cria visualização comparativa de múltiplos modelos.
    
    Parameters
    ----------
    models : dict
        Dicionário com modelos
    X_test : array-like
        Features de teste
    y_test : array-like
        Labels verdadeiros
    y_preds : dict, optional
        Predições pré-calculadas
    metric : str
        Métrica para comparar ('Acurácia', 'F1-Score (macro)', etc.)
    figsize : tuple
        Tamanho da figura
    """
    df_results = compare_models(models, X_test, y_test, y_preds)
    
    plt.figure(figsize=figsize)
    bars = plt.barh(df_results['Modelo'], df_results[metric], color='steelblue')
    
    # Adicionar valores nas barras
    for i, (bar, value) in enumerate(zip(bars, df_results[metric])):
        plt.text(value + 0.001, i, f'{value:.4f}', 
                va='center', fontweight='bold')
    
    plt.xlabel(metric, fontsize=12)
    plt.title(f'Comparação de Modelos - {metric}', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "Modelo",
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[pd.DataFrame]:
    """
    Plota importância das features (se o modelo suportar).
    
    Parameters
    ----------
    model : sklearn model
        Modelo treinado
    feature_names : list
        Nomes das features
    model_name : str
        Nome do modelo
    top_n : int, optional
        Número de features top para mostrar
    figsize : tuple
        Tamanho da figura
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame com importâncias ou None se não suportado
    """
    # Verificar se o modelo tem feature_importances_ ou coef_
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Para modelos lineares, usar valor absoluto da média dos coeficientes
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print(f"AVISO: Modelo {model_name} não suporta visualização de importância de features.")
        return None
    
    # Criar DataFrame
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importância': importances
    }).sort_values('Importância', ascending=False)
    
    if top_n:
        df_importance = df_importance.head(top_n)
    
    # Plotar
    plt.figure(figsize=figsize)
    bars = plt.barh(df_importance['Feature'], df_importance['Importância'], 
                    color='coral')
    
    # Adicionar valores
    for i, (bar, value) in enumerate(zip(bars, df_importance['Importância'])):
        plt.text(value + 0.001, i, f'{value:.4f}', 
                va='center', fontweight='bold')
    
    plt.xlabel('Importância', fontsize=12)
    plt.title(f'Importância das Features - {model_name}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return df_importance


def generate_evaluation_report(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    model_name: str = "Modelo",
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    save_plots: bool = False,
    output_dir: str = "."
) -> Dict[str, Any]:
    """
    Gera relatório completo de avaliação do modelo.
    
    Parameters
    ----------
    model : sklearn model
        Modelo treinado
    X_test : array-like
        Features de teste
    y_test : array-like
        Labels verdadeiros
    y_pred : array-like, optional
        Predições
    model_name : str
        Nome do modelo
    class_names : list, optional
        Nomes das classes
    feature_names : list, optional
        Nomes das features (para importância)
    save_plots : bool
        Se True, salva os gráficos
    output_dir : str
        Diretório para salvar gráficos
        
    Returns
    -------
    dict
        Dicionário com todas as métricas e resultados
    """
    if y_pred is None:
        y_pred = model.predict(X_test)
    
    if class_names is None:
        class_names = sorted(np.unique(y_test))
    
    print("\n" + "=" * 80)
    print(f"RELATÓRIO COMPLETO DE AVALIAÇÃO - {model_name}")
    print("=" * 80)
    
    # 1. Métricas principais
    metrics = evaluate_model(model, X_test, y_test, y_pred, model_name, verbose=True)
    
    # 2. Matriz de confusão
    print("\n")
    plot_confusion_matrix(y_test, y_pred, class_names, model_name)
    
    # 3. Relatório de classificação
    print("\n")
    print_classification_report(y_test, y_pred, class_names, model_name)
    
    # 4. Importância de features (se suportado)
    if feature_names:
        print("\n")
        df_importance = plot_feature_importance(model, feature_names, model_name)
    else:
        df_importance = None
    
    # 5. Resumo executivo
    print("\n" + "=" * 80)
    print("RESUMO EXECUTIVO")
    print("=" * 80)
    print(f"\nAcurácia Geral: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    
    # Análise de desempenho
    if metrics['accuracy'] >= 0.95:
        performance = "EXCELENTE"
    elif metrics['accuracy'] >= 0.90:
        performance = "MUITO BOM"
    elif metrics['accuracy'] >= 0.80:
        performance = "BOM"
    else:
        performance = "PRECISA MELHORAR"
    
    print(f"\nAvaliação Geral: {performance}")
    print("=" * 80)
    
    # Compilar resultados
    results = {
        'model_name': model_name,
        'metrics': metrics,
        'performance': performance,
        'feature_importance': df_importance.to_dict() if df_importance is not None else None
    }
    
    return results




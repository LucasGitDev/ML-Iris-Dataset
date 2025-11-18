"""
Módulo de pré-processamento de dados para o projeto Iris Classification.

Este módulo fornece funções para preparar os dados antes do treinamento,
incluindo conversão de labels, separação de features/target, split de dados
e normalização.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any


def convert_species_to_numeric(y: pd.Series) -> pd.Series:
    """
    Converte as espécies de string para valores numéricos.
    
    Mapeamento:
    - Iris-setosa → 1
    - Iris-versicolor → 2
    - Iris-virginica → 3
    
    Parameters
    ----------
    y : pd.Series
        Série com os nomes das espécies
        
    Returns
    -------
    pd.Series
        Série com valores numéricos convertidos
    """
    mapping = {
        'Iris-setosa': 1,
        'Iris-versicolor': 2,
        'Iris-virginica': 3
    }
    
    return y.replace(mapping)


def prepare_data(
    df: pd.DataFrame,
    target_column: str = 'species',
    convert_target: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara os dados separando features e target.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo com features e target
    target_column : str
        Nome da coluna target
    convert_target : bool
        Se True, converte target para valores numéricos
        
    Returns
    -------
    tuple
        (X, y) onde X são as features e y é o target
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    if convert_target:
        y = convert_species_to_numeric(y)
    
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide os dados em conjuntos de treino e teste.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Proporção do conjunto de teste (padrão: 0.2 = 20%)
    random_state : int
        Seed para reprodutibilidade (padrão: 42)
    stratify : bool
        Se True, mantém proporção de classes (padrão: True)
        
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
        shuffle=True
    )
    
    return X_train, X_test, y_train, y_test


def create_scaler() -> StandardScaler:
    """
    Cria um StandardScaler para normalização de features.
    
    Returns
    -------
    StandardScaler
        Scaler configurado
    """
    return StandardScaler()


def apply_scaling(
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica normalização usando StandardScaler.
    
    O scaler é ajustado apenas nos dados de treino e depois
    aplicado nos dados de teste para evitar data leakage.
    
    Parameters
    ----------
    scaler : StandardScaler
        Scaler já criado
    X_train : pd.DataFrame
        Features de treino
    X_test : pd.DataFrame
        Features de teste
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled) como arrays numpy
    """
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


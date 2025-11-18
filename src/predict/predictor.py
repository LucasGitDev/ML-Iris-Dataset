"""
Módulo de predição para classificação de espécies de íris.

Este módulo fornece classes e funções para fazer predições usando
modelos treinados.
"""

import numpy as np
import pandas as pd
import joblib
from typing import List, Tuple, Optional, Dict, Any


class IrisPredictor:
    """
    Classe wrapper para fazer predições de espécies de íris.
    
    Esta classe facilita o carregamento de modelos salvos e
    a realização de predições com formatação adequada.
    """
    
    def __init__(self, model: Any, scaler: Optional[Any] = None):
        """
        Inicializa o predictor com modelo e scaler.
        
        Parameters
        ----------
        model : sklearn model
            Modelo treinado
        scaler : sklearn scaler, optional
            Scaler usado no pré-processamento (se houver)
        """
        self.model = model
        self.scaler = scaler
        self.species_mapping = {
            1: 'Iris-setosa',
            2: 'Iris-versicolor',
            3: 'Iris-virginica'
        }
    
    @classmethod
    def load_model(
        cls,
        model_path: str,
        scaler_path: Optional[str] = None
    ) -> 'IrisPredictor':
        """
        Carrega modelo e scaler salvos.
        
        Parameters
        ----------
        model_path : str
            Caminho para o arquivo do modelo (.pkl)
        scaler_path : str, optional
            Caminho para o arquivo do scaler (.pkl)
            
        Returns
        -------
        IrisPredictor
            Instância do predictor com modelo carregado
        """
        model = joblib.load(model_path)
        scaler = None
        
        if scaler_path:
            scaler = joblib.load(scaler_path)
        
        return cls(model, scaler)
    
    def predict(
        self,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Faz predição de espécie.
        
        Parameters
        ----------
        features : array-like
            Array com 4 features: [sepal_length, sepal_width, petal_length, petal_width]
            Pode ser uma única amostra (1D) ou múltiplas (2D)
            
        Returns
        -------
        np.ndarray
            Array com predições (valores numéricos: 1, 2 ou 3)
        """
        # Garantir formato 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Aplicar scaler se disponível
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Fazer predição
        predictions = self.model.predict(features)
        
        return predictions
    
    def predict_proba(
        self,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Retorna probabilidades por classe.
        
        Parameters
        ----------
        features : array-like
            Array com 4 features
            
        Returns
        -------
        np.ndarray
            Array com probabilidades para cada classe
        """
        # Garantir formato 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Aplicar scaler se disponível
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Verificar se modelo suporta predict_proba
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
        else:
            # Se não suporta, retornar None ou array vazio
            probabilities = None
        
        return probabilities
    
    def format_prediction(
        self,
        features: np.ndarray,
        include_proba: bool = False
    ) -> Dict[str, Any]:
        """
        Formata predição para exibição ao usuário final.
        
        Parameters
        ----------
        features : array-like
            Array com 4 features
        include_proba : bool
            Se True, inclui probabilidades na saída
            
        Returns
        -------
        dict
            Dicionário com predição formatada
        """
        prediction = self.predict(features)
        pred_value = int(prediction[0]) if prediction.ndim == 0 else int(prediction[0])
        species_name = self.species_mapping[pred_value]
        
        result = {
            'species_id': pred_value,
            'species_name': species_name,
            'prediction': prediction
        }
        
        if include_proba:
            proba = self.predict_proba(features)
            if proba is not None:
                result['probabilities'] = {
                    self.species_mapping[i+1]: float(proba[0][i])
                    for i in range(len(self.species_mapping))
                }
        
        return result


def predict_interactive(
    model: Any,
    features: List[float],
    scaler: Optional[Any] = None,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Função simples para fazer predição interativa no notebook.
    
    Parameters
    ----------
    model : sklearn model
        Modelo treinado
    features : list
        Lista com 4 valores: [sepal_length, sepal_width, petal_length, petal_width]
    scaler : sklearn scaler, optional
        Scaler usado no pré-processamento
    feature_names : list, optional
        Nomes das features para exibição
        
    Returns
    -------
    dict
        Dicionário com predição formatada
    """
    if feature_names is None:
        feature_names = [
            'sepal_length_cm',
            'sepal_width_cm',
            'petal_length_cm',
            'petal_width_cm'
        ]
    
    # Converter para array numpy
    features_array = np.array(features).reshape(1, -1)
    
    # Criar predictor temporário
    predictor = IrisPredictor(model, scaler)
    
    # Fazer predição
    result = predictor.format_prediction(features_array, include_proba=True)
    
    # Adicionar informações das features
    result['input_features'] = {
        name: float(value)
        for name, value in zip(feature_names, features)
    }
    
    return result


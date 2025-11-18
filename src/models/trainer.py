"""
Módulo de treinamento de modelos de Machine Learning.

Este módulo fornece funções para criar, treinar e comparar modelos
de classificação para o problema de classificação de espécies de íris.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, Tuple
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import com fallback para funcionar tanto em módulo quanto em notebook
try:
    from ..utils.model_evaluation import compare_models as eval_compare_models
except ImportError:
    # Fallback para quando importado do notebook
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from utils.model_evaluation import compare_models as eval_compare_models


def _ensure_numeric(y: Any) -> np.ndarray:
    """
    Garante que y seja um array numpy numérico (int64).
    
    Converte strings/objetos para números usando mapeamento ou LabelEncoder.
    
    Parameters
    ----------
    y : array-like
        Target que pode ser strings ou números
        
    Returns
    -------
    np.ndarray
        Array numpy com dtype int64
    """
    # Obter valores - sempre usar .values se disponível
    if hasattr(y, 'values'):
        y_values = y.values
    elif isinstance(y, (list, tuple)):
        y_values = np.array(y)
    else:
        y_values = y
    
    # Converter para Series do pandas
    y_series = pd.Series(y_values)
    
    # Verificar se contém strings/objetos - múltiplas verificações
    is_string = (
        y_series.dtype == object or 
        str(y_series.dtype).startswith('string') or
        y_series.dtype.name == 'object' or
        (hasattr(y_series.dtype, 'kind') and y_series.dtype.kind in ['U', 'S', 'O'])
    )
    
    # Se for string/object, mapear para números
    if is_string:
        mapping = {
            'Iris-setosa': 1,
            'Iris-versicolor': 2,
            'Iris-virginica': 3
        }
        y_series = y_series.replace(mapping)
    
    # Converter para int64 - múltiplas tentativas
    y_array = None
    try:
        # Tentativa 1: conversão direta
        y_array = y_series.astype('int64').values
    except (ValueError, TypeError):
        try:
            # Tentativa 2: float -> int
            y_array = y_series.astype('float64').astype('int64').values
        except (ValueError, TypeError):
            try:
                # Tentativa 3: usar LabelEncoder
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_array = le.fit_transform(y_series.astype(str))
            except Exception:
                # Tentativa 4: conversão manual
                y_array = np.array([int(float(x)) for x in y_series])
    
    # Garantir que é array numpy com tipo int64
    y_array = np.asarray(y_array, dtype=np.int64)
    
    # Validação final crítica
    if y_array.dtype != np.int64:
        # Forçar conversão
        y_array = y_array.astype(np.int64)
    
    # Verificar se ainda é object (não deveria acontecer)
    if y_array.dtype == object:
        raise ValueError(
            f"Falha crítica: y_train ainda é object após conversão. "
            f"Valores únicos: {np.unique(y_array[:10])}, "
            f"Tipo: {y_array.dtype}"
        )
    
    return y_array


def create_baseline(strategy: str = 'most_frequent', random_state: int = 42) -> DummyClassifier:
    """
    Cria um modelo baseline usando DummyClassifier.
    
    O baseline serve como referência mínima que qualquer modelo
    sério deve superar.
    
    Parameters
    ----------
    strategy : str
        Estratégia do DummyClassifier (padrão: 'most_frequent')
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    DummyClassifier
        Modelo baseline configurado
    """
    return DummyClassifier(strategy=strategy, random_state=random_state)


def create_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Cria dicionário com todos os modelos candidatos.
    
    Modelos incluídos:
    - DecisionTree: interpretável, não precisa normalização
    - KNeighbors: simples, precisa normalização
    - GaussianNB: rápido, não precisa normalização
    - MLPClassifier: mais complexo, precisa normalização
    - Perceptron: linear, precisa normalização
    
    Parameters
    ----------
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    dict
        Dicionário com nome do modelo como chave e modelo como valor
    """
    models = {
        'DecisionTree': DecisionTreeClassifier(
            random_state=random_state,
            max_depth=None,
            min_samples_split=2
        ),
        'KNeighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform'
        ),
        'GaussianNB': GaussianNB(),
        'MLPClassifier': MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        ),
        'Perceptron': Perceptron(
            random_state=random_state,
            max_iter=1000,
            tol=1e-3
        )
    }
    
    return models


def create_model_pipelines(
    models: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> Dict[str, Pipeline]:
    """
    Cria pipelines com StandardScaler para modelos que precisam normalização.
    
    Modelos que precisam normalização:
    - KNeighbors
    - MLPClassifier
    - Perceptron
    
    Modelos que NÃO precisam normalização:
    - DecisionTree
    - GaussianNB
    
    Parameters
    ----------
    models : dict, optional
        Dicionário de modelos. Se None, cria modelos padrão
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    dict
        Dicionário com pipelines para modelos que precisam normalização
    """
    if models is None:
        models = create_models(random_state=random_state)
    
    pipelines = {}
    
    # Modelos que precisam normalização
    models_need_scaling = ['KNeighbors', 'MLPClassifier', 'Perceptron']
    
    for model_name in models_need_scaling:
        if model_name in models:
            pipelines[model_name] = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', models[model_name])
            ])
    
    return pipelines


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "Modelo"
) -> Any:
    """
    Treina um modelo individual.
    
    Esta função assume que X_train e y_train já estão no formato correto
    (arrays numpy com tipos numéricos).
    
    Parameters
    ----------
    model : sklearn model
        Modelo a ser treinado
    X_train : np.ndarray
        Features de treino (já convertido para float64)
    y_train : np.ndarray
        Target de treino (já convertido para int64)
    model_name : str
        Nome do modelo para logging
        
    Returns
    -------
    sklearn model
        Modelo treinado
    """
    # Assumir que dados já estão convertidos
    model.fit(X_train, y_train)
    return model


def train_all_models(
    models: Dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_train_scaled: Optional[Any] = None,
    pipelines: Optional[Dict[str, Pipeline]] = None
) -> Dict[str, Any]:
    """
    Treina todos os modelos do dicionário.
    
    Modelos que precisam dados normalizados usam X_train_scaled.
    Modelos que não precisam usam X_train.
    
    Parameters
    ----------
    models : dict
        Dicionário com modelos
    X_train : array-like ou DataFrame
        Features de treino (não normalizadas)
    y_train : array-like ou Series
        Target de treino
    X_train_scaled : array-like, optional
        Features de treino normalizadas (para modelos que precisam)
    pipelines : dict, optional
        Dicionário com pipelines (modelos que precisam normalização)
        
    Returns
    -------
    dict
        Dicionário com modelos treinados
    """
    # ===== CONVERSÃO INICIAL DE DADOS =====
    print("\n" + "="*60)
    print("CONVERSÃO DE DADOS - train_all_models")
    print("="*60)
    
    # Converter y_train para numérico PRIMEIRO
    print(f"\ny_train - Tipo original: {type(y_train)}")
    if hasattr(y_train, 'values'):
        y_raw = y_train.values
        print(f"   - Tem .values: Sim")
        print(f"   - Tipo de .values: {type(y_raw)}")
        if hasattr(y_raw, 'dtype'):
            print(f"   - dtype de .values: {y_raw.dtype}")
    else:
        y_raw = y_train
        print(f"   - Tem .values: Não")
        print(f"   - Tipo direto: {type(y_raw)}")
    
    # Converter para Series e mapear strings
    y_series = pd.Series(y_raw)
    print(f"   - Após pd.Series: dtype={y_series.dtype}, shape={y_series.shape}")
    print(f"   - Primeiros valores: {y_series.head(5).tolist()}")
    
    if y_series.dtype == object or str(y_series.dtype).startswith('string'):
        print(f"   AVISO: Detectado tipo object/string - aplicando mapeamento")
        mapping = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
        y_series = y_series.replace(mapping)
        print(f"   - Após mapeamento: dtype={y_series.dtype}, valores únicos={sorted(y_series.unique())}")
    
    # Converter para int64
    try:
        y_train_array = y_series.astype('int64').values
        print(f"   Conversão direta para int64: sucesso")
    except Exception as e:
        print(f"   AVISO: Conversão direta falhou: {e}")
        print(f"   - Tentando LabelEncoder...")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_array = le.fit_transform(y_series.astype(str))
        print(f"   LabelEncoder: sucesso")
    
    y_train_array = np.asarray(y_train_array, dtype=np.int64)
    print(f"   y_train_array final: dtype={y_train_array.dtype}, shape={y_train_array.shape}")
    print(f"   - Valores únicos: {sorted(np.unique(y_train_array))}")
    
    # Converter X_train para array numpy
    print(f"\nX_train - Tipo original: {type(X_train)}")
    if hasattr(X_train, 'values'):
        X_train_array = np.asarray(X_train.values, dtype=np.float64)
        print(f"   - Tem .values: Sim, shape original: {X_train.shape}")
    else:
        X_train_array = np.asarray(X_train, dtype=np.float64)
        print(f"   - Tem .values: Não")
    
    print(f"   X_train_array final: dtype={X_train_array.dtype}, shape={X_train_array.shape}")
    
    # Converter X_train_scaled se fornecido
    if X_train_scaled is not None:
        print(f"\nX_train_scaled - Tipo original: {type(X_train_scaled)}")
        if hasattr(X_train_scaled, 'values'):
            X_train_scaled_array = np.asarray(X_train_scaled.values, dtype=np.float64)
            print(f"   - Tem .values: Sim, shape original: {X_train_scaled.shape}")
        else:
            X_train_scaled_array = np.asarray(X_train_scaled, dtype=np.float64)
            print(f"   - Tem .values: Não")
        print(f"   X_train_scaled_array final: dtype={X_train_scaled_array.dtype}, shape={X_train_scaled_array.shape}")
    else:
        X_train_scaled_array = X_train_array.copy()
        print(f"\nX_train_scaled: None, usando cópia de X_train_array")
    
    print("="*60 + "\n")
    
    # ===== TREINAMENTO DOS MODELOS =====
    print("="*60)
    print("TREINANDO MODELOS")
    print("="*60)
    
    trained_models = {}
    models_need_scaling = ['KNeighbors', 'MLPClassifier', 'Perceptron']
    
    for model_name, model in models.items():
        print(f"\nTreinando: {model_name}")
        
        if model_name in models_need_scaling:
            # Usar pipeline se disponível
            if pipelines and model_name in pipelines:
                print(f"   - Usando Pipeline (com StandardScaler)")
                print(f"   - X_train tipo: {type(X_train)}, shape: {X_train.shape if hasattr(X_train, 'shape') else 'N/A'}")
                print(f"   - y_train_array tipo: {type(y_train_array)}, dtype: {y_train_array.dtype}, shape: {y_train_array.shape}")
                # Pipeline aceita DataFrame, então passar X_train original
                # mas garantir que y_train é numérico
                pipelines[model_name].fit(X_train, y_train_array)
                trained_models[model_name] = pipelines[model_name]
                print(f"   {model_name} treinado com sucesso")
            else:
                print(f"   - Usando dados normalizados (sem Pipeline)")
                print(f"   - X_train_scaled_array dtype: {X_train_scaled_array.dtype}, shape: {X_train_scaled_array.shape}")
                print(f"   - y_train_array dtype: {y_train_array.dtype}, shape: {y_train_array.shape}")
                # Modelo simples que precisa dados normalizados
                model.fit(X_train_scaled_array, y_train_array)
                trained_models[model_name] = model
                print(f"   {model_name} treinado com sucesso")
        else:
            print(f"   - Não precisa normalização")
            print(f"   - X_train_array dtype: {X_train_array.dtype}, shape: {X_train_array.shape}")
            print(f"   - y_train_array dtype: {y_train_array.dtype}, shape: {y_train_array.shape}")
            # Modelos que não precisam normalização
            model.fit(X_train_array, y_train_array)
            trained_models[model_name] = model
            print(f"   {model_name} treinado com sucesso")
    
    print("\n" + "="*60)
    print(f"TODOS OS MODELOS TREINADOS: {list(trained_models.keys())}")
    print("="*60 + "\n")
    
    return trained_models


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test_scaled: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compara o desempenho de múltiplos modelos.
    
    Parameters
    ----------
    models : dict
        Dicionário com modelos treinados
    X_test : array-like
        Features de teste (não normalizadas)
    y_test : array-like
        Target de teste
    X_test_scaled : array-like, optional
        Features de teste normalizadas
        
    Returns
    -------
    pd.DataFrame
        DataFrame com métricas comparativas
    """
    if X_test_scaled is None:
        X_test_scaled = X_test
    
    # Garantir que y_test é numérico
    y_test_array = _ensure_numeric(y_test)
    
    # Separar modelos que precisam/não precisam normalização
    models_need_scaling = ['KNeighbors', 'MLPClassifier', 'Perceptron']
    
    # Preparar predições
    y_preds = {}
    
    for model_name, model in models.items():
        if model_name in models_need_scaling:
            # Verificar se é pipeline ou modelo simples
            if hasattr(model, 'predict') and hasattr(model, 'steps'):
                # É um pipeline
                if hasattr(X_test, 'values'):
                    X_test_for_pred = X_test.values
                else:
                    X_test_for_pred = np.asarray(X_test, dtype=np.float64)
                y_pred = model.predict(X_test_for_pred)
            else:
                # É modelo simples que precisa dados normalizados
                if hasattr(X_test_scaled, 'values'):
                    X_test_for_pred = X_test_scaled.values
                else:
                    X_test_for_pred = np.asarray(X_test_scaled, dtype=np.float64)
                y_pred = model.predict(X_test_for_pred)
        else:
            # Modelo que não precisa normalização
            if hasattr(X_test, 'values'):
                X_test_for_pred = X_test.values
            else:
                X_test_for_pred = np.asarray(X_test, dtype=np.float64)
            y_pred = model.predict(X_test_for_pred)
        
        # Garantir que predições são numéricas
        y_preds[model_name] = _ensure_numeric(y_pred)
    
    # Usar função de comparação do módulo de avaliação
    return eval_compare_models(models, X_test, y_test_array, y_preds)


def select_best_model(
    comparison_df: pd.DataFrame,
    metric: str = 'Acurácia'
) -> Tuple[str, pd.Series]:
    """
    Seleciona o melhor modelo baseado em uma métrica.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame com comparação de modelos
    metric : str
        Nome da métrica para comparar (padrão: 'Acurácia')
        
    Returns
    -------
    tuple
        (nome_do_melhor_modelo, série_com_métricas_do_melhor)
    """
    best_idx = comparison_df[metric].idxmax()
    best_model_name = comparison_df.loc[best_idx, 'Modelo']
    best_metrics = comparison_df.loc[best_idx]
    
    return best_model_name, best_metrics


def save_model(
    model: Any,
    scaler: Optional[StandardScaler] = None,
    model_name: str = 'iris_classifier',
    output_dir: str = 'models'
) -> Dict[str, str]:
    """
    Salva modelo e scaler usando joblib.
    
    Parameters
    ----------
    model : sklearn model
        Modelo treinado a ser salvo
    scaler : StandardScaler, optional
        Scaler usado (se houver)
    model_name : str
        Nome base para os arquivos
    output_dir : str
        Diretório de saída
        
    Returns
    -------
    dict
        Dicionário com caminhos dos arquivos salvos
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # Salvar modelo
    model_path = os.path.join(output_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    saved_files['model'] = model_path
    
    # Salvar scaler se fornecido
    if scaler is not None:
        scaler_path = os.path.join(output_dir, f'{model_name}_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        saved_files['scaler'] = scaler_path
    
    return saved_files


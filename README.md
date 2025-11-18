# AG002 - Classificação de Espécies de Íris

Repositório para projeto de Machine Learning de classificação de espécies de íris usando o dataset Iris.

## Objetivo

Desenvolver uma pipeline completa de ML para classificar automaticamente espécies de íris a partir de medidas da flor, demonstrando todo o ciclo de vida de um projeto de Machine Learning - da ingestão de dados até a interface de predição.

## Sobre o Projeto

Este projeto implementa uma solução completa de classificação multiclasse para identificar espécies de íris (Iris-setosa, Iris-versicolor, Iris-virginica) baseado em 4 medidas:
- Comprimento da sépala (sepal_length_cm)
- Largura da sépala (sepal_width_cm)
- Comprimento da pétala (petal_length_cm)
- Largura da pétala (petal_width_cm)

## Setup

### Pré-requisitos

- Python 3.10+
- Poetry (gerenciador de dependências)

### Instalação

```bash
# Instalar dependências
poetry install

# Ativar ambiente virtual
poetry shell

# Iniciar Jupyter Notebook
poetry run jupyter notebook
```

## Estrutura do Projeto

```
AG002/
├── notebooks/
│   └── 00_template.ipynb    # Notebook principal com toda a pipeline
├── data/
│   ├── raw/                 # Dados brutos (iris.csv)
│   └── processed/           # Dados processados
├── src/
│   ├── data/                # Módulo de pré-processamento
│   │   └── preprocessing.py
│   ├── models/              # Módulo de modelos e treinamento
│   │   ├── trainer.py       # Funções de treinamento
│   │   └── comparison.py    # Comparação de abordagens
│   ├── predict/             # Módulo de predição
│   │   └── predictor.py    # Classe e funções de predição
│   └── utils/               # Utilitários
│       └── model_evaluation.py  # Funções de avaliação
├── models/                  # Modelos treinados salvos
└── .docs/                    # Documentação do projeto
```

## Como Usar

### Executar o Notebook Completo

1. Abra o Jupyter Notebook:
```bash
poetry run jupyter notebook
```

2. Navegue até `notebooks/00_template.ipynb`

3. Execute todas as células sequencialmente. O notebook contém:
   - **Seção 1-3**: Setup, carregamento de dados e EDA completo
   - **Seção 4**: Pré-processamento (conversão de species, split, normalização)
   - **Seção 5**: Estratégia de modelagem (baseline, modelos candidatos)
   - **Seção 6**: Treinamento, comparação de modelos e seleção do melhor
   - **Seção 7**: Interface de predição interativa
   - **Seção 8**: Conclusões e decisões técnicas

### Usar os Módulos Programaticamente

```python
from src.data import prepare_data, split_data, create_scaler, apply_scaling
from src.models import create_models, train_all_models, compare_models
from src.predict import predict_interactive

# Preparar dados
X, y = prepare_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Criar e treinar modelos
models = create_models()
trained_models = train_all_models(models, X_train, y_train, ...)

# Comparar modelos
comparison_df = compare_models(trained_models, X_test, y_test)

# Fazer predição
result = predict_interactive(model, features=[5.1, 3.5, 1.4, 0.2])
```

## Modelos Testados

O projeto compara 5 modelos diferentes:

1. **DecisionTree**: Árvore de decisão - interpretável, não precisa normalização
2. **KNeighbors**: k-Nearest Neighbors - simples, precisa normalização
3. **GaussianNB**: Naive Bayes - rápido, não precisa normalização
4. **MLPClassifier**: Multi-Layer Perceptron - complexo, precisa normalização
5. **Perceptron**: Perceptron - linear, precisa normalização

## Abordagens Comparadas

- **Abordagem A (Sem Normalização)**: DecisionTree, GaussianNB
- **Abordagem B (Com Normalização)**: KNeighbors, MLPClassifier, Perceptron

O notebook compara ambas as abordagens para identificar qual funciona melhor.

## Métricas

- **Métrica Principal**: Acurácia
- **Métricas Secundárias**: Precision, Recall, F1-Score (por classe e macro)

## Tecnologias

- **Python 3.10+**
- **Jupyter Notebook**
- **Bibliotecas**:
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - numpy

## Públicos

O projeto foi desenvolvido para atender dois públicos:

- **Técnico**: Código comentado, decisões documentadas, estrutura modular
- **Não-técnico**: Visualizações claras, resumo executivo, explicações em markdown

## Estrutura Modular

A estrutura modular permite:
- Reutilização de código em outros projetos
- Facilita criação de interfaces visuais futuras
- Manutenção e extensão simplificadas

## Próximos Passos

Melhorias futuras planejadas:
- Validação cruzada (k-fold)
- Tuning de hiperparâmetros
- Feature engineering
- Interface web (Streamlit/Flask)
- Análise de importância de features


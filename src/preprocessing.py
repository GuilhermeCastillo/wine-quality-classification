"""
Divisão treino/teste e padronização.

A ordem das operações aqui é o que previne vazamento de dados:
primeiro separa, depois padroniza  e a padronização vive dentro
de um Pipeline, para que média e desvio sejam calculados apenas
sobre o treino e reaplicados ao teste.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Separa treino e teste preservando a proporção das classes.

    `stratify=y` é obrigatório aqui: com apenas ~14% de vinhos de
    alta qualidade, um sorteio sem estratificação poderia gerar
    conjuntos com proporções muito diferentes e enviesar a avaliação.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def criar_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Cria o esquema de validação cruzada estratificada.

    Centralizar aqui garante que todos os experimentos do projeto
    usem exatamente as mesmas dobras — pré-requisito para que as
    comparações entre modelos sejam justas.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def com_padronizacao(estimador) -> Pipeline:
    """Envolve um estimador num Pipeline com StandardScaler.

    Necessário para modelos sensíveis à escala (Regressão Logística,
    SVM, KNN). Modelos baseados em árvore não precisam: cortes por
    limiar são invariantes à escala das variáveis.
    """
    return Pipeline([("scaler", StandardScaler()), ("clf", estimador)])

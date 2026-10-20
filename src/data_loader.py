"""
Carregamento da base de vinhos e criação da variável alvo.

Este módulo isola tudo que diz respeito à leitura do dado bruto.
Nenhuma outra parte do projeto deve ler o CSV diretamente.
"""

from __future__ import annotations

import pandas as pd

# Coluna de nota original atribuída pelo especialista
COLUNA_NOTA = "quality"

# Coluna binária derivada: 1 = alta qualidade, 0 = baixa/média
COLUNA_ALVO = "high_quality"

# Colunas que não são medições físico-químicas e devem sair do modelo
COLUNAS_DESCARTAVEIS = ["Id"]


def load_wine_data(caminho: str, limiar_nota: int = 7) -> pd.DataFrame:
    """Carrega o CSV e cria a variável alvo binária.

    Parâmetros
    ----------
    caminho : str
        Caminho do arquivo CSV (ex.: "../data/WineQT.csv").
    limiar_nota : int
        Nota mínima para o vinho ser considerado de alta qualidade.
        O enunciado do desafio define 7.

    Retorna
    -------
    pd.DataFrame
        Base com as 11 variáveis físico-químicas, a nota original
        e a coluna binária `high_quality`.
    """
    df = pd.read_csv(caminho)

    # `Id` é apenas identificador de linha; manter criaria correlação
    # espúria com a ordem dos registros.
    descartar = [c for c in COLUNAS_DESCARTAVEIS if c in df.columns]
    if descartar:
        df = df.drop(columns=descartar)

    df[COLUNA_ALVO] = (df[COLUNA_NOTA] >= limiar_nota).astype(int)
    return df


def separar_features_alvo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X e y, removendo a nota original de X.

    `quality` PRECISA sair das features: `high_quality` é derivada
    dela, então mantê-la causaria vazamento de alvo — o modelo
    simplesmente reaprenderia a regra `quality >= 7`.
    """
    X = df.drop(columns=[COLUNA_NOTA, COLUNA_ALVO])
    y = df[COLUNA_ALVO]
    return X, y


def resumo_base(df: pd.DataFrame) -> dict:
    """Devolve um resumo diagnóstico da base, útil na EDA e no README."""
    n_total = len(df)
    n_alta = int(df[COLUNA_ALVO].sum())
    return {
        "linhas": n_total,
        "colunas": df.shape[1],
        "valores_nulos": int(df.isna().sum().sum()),
        "linhas_duplicadas": int(df.duplicated().sum()),
        "vinhos_alta_qualidade": n_alta,
        "proporcao_alta_qualidade": round(n_alta / n_total, 4),
        "baseline_acuracia_trivial": round(1 - n_alta / n_total, 4),
    }

"""
Tradução dos fatores do modelo para linguagem de negócio.

A importância que a Random Forest calcula diz o quanto cada variável
pesa no modelo como um todo, mas não explica uma amostra específica.
Aqui combinamos três informações para explicar vinho a vinho:

1. o peso geral da variável (importância do modelo);
2. a posição daquela amostra na distribuição histórica (percentil);
3. a direção do efeito (se valores altos ajudam ou atrapalham).

O resultado é aproximado — não substitui SHAP —, mas é transparente,
não exige dependência adicional e é suficiente para uso executivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Direção do efeito de cada variável, obtida dos odds ratios da
# Regressão Logística e coerente com a literatura enológica.
# +1 = valores altos favorecem a qualidade; -1 = prejudicam.
DIRECAO = {
    "alcohol": 1,
    "sulphates": 1,
    "citric acid": 1,
    "fixed acidity": 1,
    "residual sugar": 1,
    "free sulfur dioxide": 1,
    "pH": -1,
    "density": -1,
    "chlorides": -1,
    "volatile acidity": -1,
    "total sulfur dioxide": -1,
}

# Nomes em linguagem acessível, para exibição ao usuário final.
NOMES_AMIGAVEIS = {
    "fixed acidity": "Acidez fixa",
    "volatile acidity": "Acidez volátil",
    "citric acid": "Ácido cítrico",
    "residual sugar": "Açúcar residual",
    "chlorides": "Teor de sal",
    "free sulfur dioxide": "Enxofre livre",
    "total sulfur dioxide": "Enxofre total",
    "density": "Densidade",
    "pH": "pH",
    "sulphates": "Sulfatos",
    "alcohol": "Teor alcoólico",
}

# Explicação enológica de cada fator, usada nos textos do relatório.
CONTEXTO = {
    "alcohol": (
        "teor alcoólico elevado indica uvas colhidas no ponto certo de maturação",
        "teor alcoólico baixo sugere uvas menos maduras",
    ),
    "volatile acidity": (
        "baixa acidez volátil indica fermentação bem conduzida",
        "acidez volátil alta é o principal defeito de fermentação, percebido como aroma de vinagre",
    ),
    "sulphates": (
        "nível adequado de sulfatos protege o vinho da oxidação",
        "sulfatos abaixo do usual deixam o vinho mais exposto à oxidação",
    ),
    "citric acid": (
        "o ácido cítrico contribui para a sensação de frescor",
        "pouco ácido cítrico deixa o vinho menos vivo no paladar",
    ),
    "total sulfur dioxide": (
        "enxofre total controlado preserva o aroma",
        "excesso de enxofre total pode mascarar os aromas do vinho",
    ),
    "chlorides": (
        "teor de sal dentro do esperado",
        "teor de sal acima do usual pode deixar o vinho com sensação salina",
    ),
    "density": (
        "densidade compatível com vinhos de maior teor alcoólico",
        "densidade elevada costuma acompanhar menor teor alcoólico",
    ),
    "fixed acidity": (
        "boa estrutura ácida, que dá sustentação ao vinho",
        "acidez fixa baixa pode deixar o vinho mole",
    ),
    "residual sugar": (
        "açúcar residual dentro do perfil usual",
        "açúcar residual fora do perfil usual",
    ),
    "free sulfur dioxide": (
        "enxofre livre em nível protetor",
        "enxofre livre baixo reduz a proteção contra oxidação",
    ),
    "pH": (
        "pH equilibrado, indicando boa acidez",
        "pH alto indica acidez insuficiente",
    ),
}


def calcular_referencia(df_treino: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Calcula as estatísticas históricas usadas como base de comparação."""
    return df_treino[features].describe().T[["mean", "std", "25%", "50%", "75%"]]


def _percentil(valor: float, serie: pd.Series) -> float:
    """Posição do valor na distribuição histórica, de 0 a 1."""
    return float((serie <= valor).mean())


def explicar_amostra(
    amostra: pd.Series,
    importancias: pd.Series,
    df_referencia: pd.DataFrame,
    n_fatores: int = 3,
) -> dict:
    """Explica uma amostra individual em linguagem de negócio.

    Retorna os fatores que mais puxaram a avaliação para cima e para
    baixo, com o texto já pronto para exibição.
    """
    contribuicoes = []

    for feature, peso in importancias.items():
        if feature not in amostra.index or feature not in DIRECAO:
            continue

        valor = float(amostra[feature])
        p = _percentil(valor, df_referencia[feature])

        # Desvio em relação à mediana histórica, de -1 a +1.
        desvio = (p - 0.5) * 2
        contribuicao = peso * desvio * DIRECAO[feature]

        contribuicoes.append(
            {
                "variavel": feature,
                "nome": NOMES_AMIGAVEIS.get(feature, feature),
                "valor": valor,
                "percentil": round(p * 100),
                "contribuicao": contribuicao,
            }
        )

    contribuicoes.sort(key=lambda d: d["contribuicao"], reverse=True)

    positivos = [c for c in contribuicoes if c["contribuicao"] > 0][:n_fatores]
    negativos = [c for c in contribuicoes if c["contribuicao"] < 0][-n_fatores:][::-1]

    return {
        "pontos_fortes": [_texto(c, favoravel=True) for c in positivos],
        "pontos_fracos": [_texto(c, favoravel=False) for c in negativos],
        "detalhe": contribuicoes,
    }


def _texto(c: dict, favoravel: bool) -> str:
    """Monta a frase explicativa de um fator."""
    idx = 0 if favoravel else 1
    contexto = CONTEXTO.get(c["variavel"], ("", ""))[idx]
    p = c["percentil"]

    # Só afirmamos o efeito enológico quando o valor é de fato atípico.
    # Dentro da faixa usual, a contribuição existe mas é fraca, e uma
    # frase categórica soaria mais confiante do que o dado permite.
    if p >= 75:
        return f"{c['nome']} de {c['valor']:.2f}, acima da maioria dos vinhos analisados: {contexto}."
    if p <= 25:
        return f"{c['nome']} de {c['valor']:.2f}, abaixo da maioria dos vinhos analisados: {contexto}."

    lado = "levemente a favor" if favoravel else "levemente contra"
    return f"{c['nome']} de {c['valor']:.2f}, dentro da faixa usual, pesando {lado}."


def resumo_executivo(prob: float, limiar: float, explicacao: dict) -> str:
    """Gera um parágrafo de conclusão para a amostra."""
    if prob >= limiar:
        abertura = (
            f"Este vinho tem {prob*100:.0f}% de probabilidade de ser avaliado como "
            "alta qualidade e foi recomendado para prova sensorial prioritária."
        )
    elif prob >= limiar * 0.75:
        abertura = (
            f"Este vinho ficou perto do limite de recomendação ({prob*100:.0f}%). "
            "Vale uma segunda análise antes de descartar."
        )
    else:
        abertura = (
            f"Este vinho tem {prob*100:.0f}% de probabilidade de alta qualidade, "
            "abaixo do ponto de corte definido."
        )

    if explicacao["pontos_fortes"]:
        abertura += " O que pesou a favor: " + explicacao["pontos_fortes"][0].lower()
    if explicacao["pontos_fracos"]:
        abertura += " O que pesou contra: " + explicacao["pontos_fracos"][0].lower()

    return abertura


def classificar_lote(pacote: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Classifica um lote inteiro de vinhos.

    Retorna o dataframe original acrescido da probabilidade, da
    classificação e da posição no ranking.
    """
    features = pacote["features"]
    faltando = [f for f in features if f not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no arquivo: {', '.join(faltando)}")

    X = df[features]
    prob = pacote["modelo"].predict_proba(X)[:, 1]

    saida = df.copy()
    saida["probabilidade"] = prob
    saida["classificacao"] = np.where(
        prob >= pacote["limiar"], "Alta qualidade", "Baixa/Média qualidade"
    )
    saida["ranking"] = (
        saida["probabilidade"].rank(ascending=False, method="min").astype(int)
    )
    return saida.sort_values("probabilidade", ascending=False)


def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Ajusta os nomes das colunas do arquivo enviado aos nomes canônicos.

    Tolera variações comuns de digitação (maiúsculas, underscore, espaços
    extras) sem alterar a grafia esperada pelo modelo. Importante: não
    basta converter tudo para minúsculas, porque `pH` perderia a grafia
    correta e deixaria de ser reconhecida.
    """

    def chave(nome: str) -> str:
        return nome.strip().lower().replace("_", " ").replace("-", " ")

    canonicos = {chave(f): f for f in DIRECAO}
    renomear = {col: canonicos.get(chave(col), col) for col in df.columns}
    return df.rename(columns=renomear)

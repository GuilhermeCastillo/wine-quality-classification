"""
Definição, treino, ajuste de limiar e persistência dos modelos.

O limiar de decisão é tratado aqui como parte do modelo, e não como
detalhe de avaliação: ele é escolhido no treino (via validação
cruzada), salvo junto com o estimador e reaplicado em produção.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_predict

from .preprocessing import RANDOM_STATE, com_padronizacao

# Hiperparâmetros validados por GridSearchCV (108 combinações, scoring=f1).
# A busca mostrou ganho dentro do ruído frente aos valores padrão; estes
# foram mantidos por serem mais regularizados (profundidade limitada e
# mínimo de amostras por folha), o que é prudente numa base com poucos
# exemplos da classe positiva.
RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 10,
    "max_features": "sqrt",
    "min_samples_leaf": 4,
    "class_weight": "balanced",
}


def criar_modelos(random_state: int = RANDOM_STATE) -> dict:
    """Devolve os modelos candidatos do projeto.

    A Regressão Logística vai embrulhada em Pipeline com scaler;
    a Random Forest, não. Isso não é esquecimento: árvores são
    invariantes à escala.
    """
    return {
        "Regressão Logística": com_padronizacao(
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state
            )
        ),
        "Random Forest": RandomForestClassifier(
            random_state=random_state, n_jobs=-1, **RF_PARAMS
        ),
    }


def buscar_hiperparametros(X_train, y_train, cv, random_state: int = RANDOM_STATE):
    """Executa a busca em grade da Random Forest.

    Roda apenas sobre o TREINO: usar o conjunto de teste para
    escolher hiperparâmetros tornaria a avaliação final otimista.
    O scoring é F1 e não acurácia — otimizar acurácia levaria a
    busca a favorecer modelos que ignoram a classe rara.
    """
    grade = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
    busca = GridSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        grade,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    busca.fit(X_train, y_train)
    return busca


def encontrar_limiar_otimo(modelo, X_train, y_train, cv) -> dict:
    """Escolhe o limiar de decisão que maximiza o F1.

    As probabilidades vêm de `cross_val_predict`: cada amostra é
    prevista por um modelo que não a viu durante o treino. Escolher
    o limiar diretamente sobre o conjunto de teste inflaria o
    resultado, porque seria ajustar à prova que ainda será aplicada.
    """
    y_prob = cross_val_predict(modelo, X_train, y_train, cv=cv, method="predict_proba")[
        :, 1
    ]

    precisao, recall, limiares = precision_recall_curve(y_train, y_prob)
    f1 = 2 * (precisao * recall) / (precisao + recall + 1e-9)

    idx = int(np.argmax(f1[:-1]))
    return {
        "limiar": float(limiares[idx]),
        "f1_esperado": float(f1[idx]),
        "precisao_esperada": float(precisao[idx]),
        "recall_esperado": float(recall[idx]),
        "curva": {
            "limiares": limiares,
            "precisao": precisao[:-1],
            "recall": recall[:-1],
            "f1": f1[:-1],
        },
    }


def prever_com_limiar(modelo, X, limiar: float = 0.5) -> np.ndarray:
    """Classifica usando um limiar explícito em vez do 0.5 implícito.

    O 0.5 embutido no `.predict()` do scikit-learn é convenção
    herdada de problemas balanceados, não uma escolha fundamentada.
    """
    return (modelo.predict_proba(X)[:, 1] >= limiar).astype(int)


def salvar_modelo(modelo, limiar: float, features: list[str], caminho: str) -> None:
    """Persiste modelo, limiar e ordem das features num único arquivo.

    As três coisas andam juntas de propósito. Sem o limiar, quem
    carregar o modelo usaria 0.5 e perderia o ganho do ajuste. Sem
    a ordem das features, um formulário que monte o vetor em ordem
    diferente produziria previsões erradas sem lançar erro algum.
    """
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"modelo": modelo, "limiar": float(limiar), "features": list(features)},
        caminho,
    )


def carregar_modelo(caminho: str) -> dict:
    """Carrega o pacote salvo por `salvar_modelo`."""
    return joblib.load(caminho)


def classificar_vinho(pacote: dict, amostra: dict) -> dict:
    """Classifica uma única amostra a partir de um dicionário de medições.

    É o ponto de entrada pensado para o app Streamlit. Reordena as
    variáveis conforme o treino e aplica o limiar salvo.
    """
    faltando = [f for f in pacote["features"] if f not in amostra]
    if faltando:
        raise ValueError(f"Variáveis ausentes na amostra: {faltando}")

    X = pd.DataFrame([{f: amostra[f] for f in pacote["features"]}])
    prob = float(pacote["modelo"].predict_proba(X)[:, 1][0])
    return {
        "probabilidade_alta_qualidade": round(prob, 4),
        "classificacao": (
            "Alta qualidade" if prob >= pacote["limiar"] else "Baixa/Média qualidade"
        ),
        "limiar_utilizado": pacote["limiar"],
    }

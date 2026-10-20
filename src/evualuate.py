"""
Avaliação dos modelos e geração dos gráficos de resultado.

Nenhuma função aqui imprime: todas retornam dados ou salvam
arquivos. Quem chama decide como exibir.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from .models import prever_com_limiar

NOMES_CLASSES = ["Baixa/Média", "Alta"]


def _salvar(fig, caminho: str | None) -> None:
    if caminho:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, dpi=150, bbox_inches="tight")


def avaliar_modelo(modelo, X_test, y_test, limiar: float = 0.5) -> dict:
    """Calcula as métricas da classe positiva sobre o conjunto de teste.

    O foco é a classe "Alta": é a rara e a que interessa ao negócio.
    A acurácia é incluída apenas para contraste com o baseline trivial.
    """
    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "limiar": round(float(limiar), 4),
        "acuracia": round(float((tp + tn) / len(y_test)), 4),
        "precisao_alta": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall_alta": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_alta": round(f1_score(y_test, y_pred, zero_division=0), 4),
        # O ROC-AUC não depende do limiar: mede a capacidade de ordenar,
        # não o ponto de corte.
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "verdadeiros_positivos": int(tp),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdadeiros_negativos": int(tn),
    }


def comparar_modelos_cv(modelos: dict, X_train, y_train, cv) -> pd.DataFrame:
    """Compara modelos por validação cruzada sobre o TREINO.

    Usar a base completa aqui incluiria o conjunto de teste na
    validação e deixaria de ser uma estimativa limpa.
    """
    linhas = []
    for nome, modelo in modelos.items():
        f1 = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1")
        auc = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc")
        linhas.append(
            {
                "modelo": nome,
                "f1_medio": round(f1.mean(), 4),
                "f1_desvio": round(f1.std(), 4),
                "auc_medio": round(auc.mean(), 4),
                "auc_desvio": round(auc.std(), 4),
            }
        )
    return pd.DataFrame(linhas).sort_values("f1_medio", ascending=False)


def plotar_matrizes_confusao(
    modelos_preditos: dict, y_test, caminho: str | None = None
):
    """Desenha lado a lado as matrizes de confusão.

    `modelos_preditos` é um dict {nome: vetor_de_predicoes}.
    """
    n = len(modelos_preditos)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2))
    axes = np.atleast_1d(axes)

    for ax, (nome, y_pred) in zip(axes, modelos_preditos.items()):
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred), display_labels=NOMES_CLASSES
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(nome)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")

    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def plotar_importancia(modelo, nomes_features, caminho: str | None = None):
    """Gráfico de importância das variáveis (modelos baseados em árvore).

    Atenção na leitura: a importância por redução de impureza
    favorece variáveis contínuas com muitos valores distintos, e
    dilui a importância entre variáveis correlacionadas entre si.
    """
    imp = pd.Series(modelo.feature_importances_, index=nomes_features).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind="barh", color="#7b1e3c", ax=ax)
    ax.set_title("Importância das variáveis")
    ax.set_xlabel("Importância relativa")
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig, imp.sort_values(ascending=False)


def plotar_curva_limiar(curva: dict, limiar: float, caminho: str | None = None):
    """Mostra como precisão, recall e F1 variam com o limiar de decisão."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curva["limiares"], curva["precisao"], label="Precisão")
    ax.plot(curva["limiares"], curva["recall"], label="Recall")
    ax.plot(curva["limiares"], curva["f1"], label="F1", linestyle="--")
    ax.axvline(limiar, color="red", alpha=0.6, label=f"Limiar escolhido = {limiar:.2f}")
    ax.axvline(0.5, color="gray", alpha=0.5, linestyle=":", label="Padrão = 0.50")
    ax.set_xlabel("Limiar de decisão")
    ax.set_ylabel("Métrica")
    ax.set_title("Trade-off precisão × recall conforme o limiar")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def plotar_boxplots_por_classe(df, features, coluna_alvo, caminho: str | None = None):
    """Painel de boxplots de cada variável, separado por classe."""
    n = len(features)
    linhas = int(np.ceil(n / 3))
    fig, axes = plt.subplots(linhas, 3, figsize=(15, 4 * linhas))

    for ax, col in zip(axes.flat, features):
        sns.boxplot(data=df, x=coluna_alvo, y=col, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(
        "Distribuição das variáveis por classe (0 = baixa/média, 1 = alta)", y=1.0
    )
    fig.tight_layout()
    _salvar(fig, caminho)
    return fig


def contar_outliers(df: pd.DataFrame, features) -> pd.Series:
    """Conta outliers pelo critério do intervalo interquartil (1.5 × IQR).

    O critério pressupõe distribuição simétrica. Concentrações
    químicas são assimétricas à direita por natureza, então parte
    do que é marcado aqui é artefato do método, não anomalia real.
    """
    q1, q3 = df[features].quantile(0.25), df[features].quantile(0.75)
    iqr = q3 - q1
    fora = (df[features] < q1 - 1.5 * iqr) | (df[features] > q3 + 1.5 * iqr)
    return fora.sum().sort_values(ascending=False)

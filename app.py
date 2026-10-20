"""
Classificador de qualidade de vinhos.

Aplicação Streamlit que recebe um arquivo CSV com as medições
físico-químicas de um lote e indica quais amostras têm maior
probabilidade de serem avaliadas como alta qualidade.

Execução local:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.explain import (
    NOMES_AMIGAVEIS,
    classificar_lote,
    explicar_amostra,
    normalizar_colunas,
    resumo_executivo,
)
from src.models import carregar_modelo

RAIZ = Path(__file__).parent
CAMINHO_MODELO = RAIZ / "models" / "random_forest.pkl"
CAMINHO_DADOS = RAIZ / "data" / "WineQT.csv"

st.set_page_config(
    page_title="Classificador de Qualidade de Vinhos",
    page_icon=None,
    layout="wide",
)


@st.cache_resource
def carregar_pacote():
    if not CAMINHO_MODELO.exists():
        return None
    return carregar_modelo(str(CAMINHO_MODELO))


@st.cache_data
def carregar_referencia(features):
    """Base histórica usada para comparar cada amostra."""
    df = pd.read_csv(CAMINHO_DADOS)
    return df[features]


@st.cache_data
def carregar_importancias(_modelo, features):
    return pd.Series(_modelo.feature_importances_, index=features).sort_values(
        ascending=False
    )


pacote = carregar_pacote()

if pacote is None:
    st.error(
        "Modelo não encontrado. Execute o notebook `02_modelagem.ipynb` "
        "para gerar o arquivo `models/random_forest.pkl`."
    )
    st.stop()

features = pacote["features"]
limiar = pacote["limiar"]
df_referencia = carregar_referencia(features)
importancias = carregar_importancias(pacote["modelo"], features)


# ----------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Como funciona")
    st.write(
        "O modelo analisa 11 medições de laboratório e estima a chance de cada "
        "vinho ser avaliado como alta qualidade por um especialista."
    )
    st.write(
        "A recomendação não substitui a prova sensorial. Ela serve para "
        "priorizar quais lotes merecem avaliação, reduzindo tempo e custo."
    )

    st.divider()
    st.subheader("Ponto de corte")
    st.write(
        f"Vinhos com probabilidade acima de **{limiar*100:.0f}%** são recomendados. "
        "Esse valor foi calibrado para reduzir recomendações equivocadas sem "
        "deixar bons vinhos de fora."
    )

    st.divider()
    st.subheader("Arquivo esperado")
    st.write("Um CSV com uma linha por vinho e as seguintes colunas:")
    st.code("\n".join(features), language=None)

    modelo_csv = pd.DataFrame(columns=features)
    st.download_button(
        "Baixar modelo de planilha",
        modelo_csv.to_csv(index=False).encode("utf-8"),
        "modelo_entrada.csv",
        "text/csv",
        use_container_width=True,
    )


# ----------------------------------------------------------------------
# Conteúdo principal
# ----------------------------------------------------------------------
st.title("Classificador de Qualidade de Vinhos")
st.write(
    "Envie as medições de laboratório de um lote e descubra quais vinhos têm "
    "maior potencial de alta qualidade, com a justificativa de cada indicação."
)

arquivo = st.file_uploader("Arquivo CSV com as medições", type=["csv"])

usar_exemplo = st.checkbox("Usar uma amostra de exemplo da base histórica")

df_entrada = None

if arquivo is not None:
    try:
        df_entrada = normalizar_colunas(pd.read_csv(arquivo))
    except Exception as erro:
        st.error(f"Não foi possível ler o arquivo: {erro}")
elif usar_exemplo:
    df_entrada = (
        pd.read_csv(CAMINHO_DADOS).sample(25, random_state=7).reset_index(drop=True)
    )
    df_entrada = normalizar_colunas(df_entrada)

if df_entrada is None:
    st.info("Envie um arquivo ou marque a opção de amostra de exemplo para começar.")
    st.stop()

try:
    resultado = classificar_lote(pacote, df_entrada)
except ValueError as erro:
    st.error(str(erro))
    st.info(
        "Confira se os nomes das colunas estão iguais aos da planilha modelo "
        "disponível na barra lateral."
    )
    st.stop()


# ----------------------------------------------------------------------
# Visão geral do lote
# ----------------------------------------------------------------------
total = len(resultado)
recomendados = int((resultado["classificacao"] == "Alta qualidade").sum())

st.subheader("Visão geral do lote")

col1, col2, col3 = st.columns(3)
col1.metric("Vinhos analisados", total)
col2.metric("Recomendados para prova", recomendados)
col3.metric(
    "Proporção do lote",
    f"{recomendados/total*100:.0f}%" if total else "0%",
)

if recomendados == 0:
    st.warning(
        "Nenhum vinho deste lote atingiu o ponto de corte. Os de maior "
        "probabilidade aparecem no topo da tabela abaixo."
    )
else:
    st.success(
        f"{recomendados} de {total} vinhos foram indicados para avaliação sensorial "
        "prioritária."
    )


# ----------------------------------------------------------------------
# Tabela de resultados
# ----------------------------------------------------------------------
st.subheader("Resultado por vinho")

colunas_exibir = ["ranking", "classificacao", "probabilidade"] + features
tabela = resultado[colunas_exibir].rename(
    columns={
        "ranking": "Posição",
        "classificacao": "Avaliação",
        "probabilidade": "Chance de alta qualidade",
        **NOMES_AMIGAVEIS,
    }
)

st.dataframe(
    tabela,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Chance de alta qualidade": st.column_config.ProgressColumn(
            "Chance de alta qualidade",
            format="%.0f%%",
            min_value=0,
            max_value=1,
        )
    },
)

st.download_button(
    "Baixar resultado completo",
    resultado.to_csv(index=False).encode("utf-8"),
    "resultado_classificacao.csv",
    "text/csv",
)


# ----------------------------------------------------------------------
# Explicação individual
# ----------------------------------------------------------------------
st.subheader("Por que este vinho foi indicado")

opcoes = {
    f"Posição {int(linha['ranking'])} — {linha['probabilidade']*100:.0f}% de chance": idx
    for idx, linha in resultado.iterrows()
}
escolha = st.selectbox("Escolha um vinho para ver a justificativa", list(opcoes.keys()))
idx = opcoes[escolha]
amostra = resultado.loc[idx]

explicacao = explicar_amostra(amostra, importancias, df_referencia)

st.write(resumo_executivo(amostra["probabilidade"], limiar, explicacao))

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Pontos a favor**")
    if explicacao["pontos_fortes"]:
        for texto in explicacao["pontos_fortes"]:
            st.write(f"- {texto}")
    else:
        st.write("Nenhum fator relevante a favor.")

with col_b:
    st.markdown("**Pontos de atenção**")
    if explicacao["pontos_fracos"]:
        for texto in explicacao["pontos_fracos"]:
            st.write(f"- {texto}")
    else:
        st.write("Nenhum fator relevante contra.")


# ----------------------------------------------------------------------
# Fatores gerais
# ----------------------------------------------------------------------
with st.expander("O que mais influencia a qualidade, de modo geral"):
    st.write(
        "Estes são os fatores que mais pesam na avaliação, considerando todos os "
        "vinhos analisados no estudo:"
    )
    ranking = importancias.head(5)
    for nome, peso in ranking.items():
        st.write(
            f"- **{NOMES_AMIGAVEIS.get(nome, nome)}**: {peso*100:.0f}% de influência"
        )

    st.caption(
        "Teor alcoólico e acidez volátil concentram a maior parte da explicação. "
        "O primeiro indica uvas colhidas no ponto certo; a segunda é o principal "
        "defeito de fermentação e é o fator mais controlável no processo produtivo."
    )

st.caption(
    "Modelo treinado com 1.143 amostras de vinho tinto. As indicações são "
    "estatísticas e complementam, não substituem, a avaliação de especialistas."
)

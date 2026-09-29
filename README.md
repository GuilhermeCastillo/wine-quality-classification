# Wine Quality Classification

Projeto desenvolvido para classificar vinhos em duas categorias de qualidade
a partir de características físico-químicas.

## Objetivo

Prever se um vinho pertence à categoria de alta qualidade ou baixa/média qualidade.

A classificação será definida da seguinte forma:

- Alta qualidade: quality >= 7
- Baixa/média qualidade: quality < 7

## Estrutura do projeto

- `data/`: dados brutos, intermediários e processados.
- `notebooks/`: análises exploratórias e experimentos.
- `src/`: código reutilizável da pipeline.
- `results/`: gráficos, métricas e modelos.
- `presentation/`: apresentação executiva.
- `docs/`: documentação do projeto.
- `tests/`: testes básicos.

## Modelos

Serão avaliados pelo menos dois modelos de classificação:

- Regressão Logística.
- Random Forest.

## Métricas

Os modelos serão comparados utilizando:

- Acurácia.
- Precisão.
- Recall.
- F1-score.
- Matriz de confusão.

# Classificação da Qualidade de Vinhos com Machine Learning

Tech Challenge Fase 2, FIAP POSTECH.

O projeto desenvolve um modelo de classificação que prevê se um vinho é de alta qualidade a partir de suas características físico-químicas.

## Contexto

A qualidade de um vinho é avaliada por especialistas através de análise sensorial. O processo é subjetivo, demorado e depende da disponibilidade de profissionais experientes.

Durante a produção, o laboratório já mede variáveis objetivas como acidez, teor alcoólico, densidade e níveis de dióxido de enxofre. A questão investigada aqui é se essas medições conseguem antecipar a nota que um especialista daria.

Em caso positivo, o produtor passa a ter um instrumento de triagem capaz de priorizar quais lotes merecem avaliação sensorial e de sinalizar desvios de processo antes do engarrafamento.

## Dados

Fonte: Wine Quality Dataset, disponível no Kaggle (arquivo `WineQT.csv`).

A base contém 1.143 amostras de vinho tinto, com 11 variáveis preditoras e uma nota de qualidade de 3 a 8 atribuída por especialistas. Não há valores nulos nem linhas duplicadas.

O escopo foi limitado ao vinho tinto. O dataset de vinhos brancos tem perfil físico-químico distinto e sua inclusão exigiria tratamento separado por tipo.

Vale registrar que o dataset original do UCI contém 1.599 linhas de vinho tinto, com centenas de duplicatas. A versão utilizada já vem sem essas repetições, o que foi verificado na análise.

### Variável alvo

Conforme o enunciado, a nota foi convertida em classificação binária:

| Classe                | Critério                | Amostras | Proporção |
| --------------------- | ----------------------- | -------- | --------- |
| Alta qualidade        | nota maior ou igual a 7 | 159      | 13,9%     |
| Baixa/média qualidade | nota menor que 7        | 984      | 86,1%     |

Essa distribuição tem uma consequência importante. Um modelo que classifique todas as amostras como baixa/média qualidade, sem analisar variável alguma, atinge 86,1% de acurácia. Esse é o baseline trivial contra o qual qualquer resultado precisa ser comparado, e é a razão pela qual a acurácia não foi adotada como métrica principal.

## Estrutura do repositório

```
wine-quality-classification/
│
├── data/
│   └── WineQT.csv                 # base original, não modificada
│
├── notebooks/
│   ├── 01_eda.ipynb               # análise exploratória
│   └── 02_modelagem.ipynb         # modelagem, avaliação e interpretação
│
├── src/
│   ├── data_loader.py             # carregamento e criação da variável alvo
│   ├── preprocessing.py           # divisão treino/teste e padronização
│   ├── models.py                  # definição, treino, limiar e persistência
│   └── evaluate.py                # métricas e gráficos
│
├── models/
│   └── random_forest.pkl          # modelo treinado, limiar e ordem das features
│
├── results/                       # gráficos e métricas exportadas
│
├── requirements.txt
└── README.md
```

## Como executar

```bash
git clone https://github.com/GuilhermeCastillo/wine-quality-classification.git
cd wine-quality-classification

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Execute primeiro o notebook `01_eda.ipynb` e depois o `02_modelagem.ipynb`. Os resultados são reproduzíveis, já que a semente aleatória está fixada em 42 em todas as etapas.

## Análise exploratória

A EDA mostrou que nenhuma variável isolada tem correlação superior a 0,41 com o alvo. As mais relevantes são o teor alcoólico, com 0,40, e a acidez volátil, com -0,30.

Esse resultado orientou a modelagem. Se uma única variável determinasse a qualidade, uma regra simples resolveria o problema. A qualidade emerge da combinação de fatores, o que justifica o uso de machine learning e já indicava vantagem para modelos não lineares.

Também foi identificada multicolinearidade entre as preditoras, com destaque para acidez fixa e pH (-0,69), acidez fixa e densidade (0,68) e dióxido de enxofre livre e total (0,66). Todas têm explicação físico-química direta e foram mantidas, com a ressalva registrada na interpretação dos resultados.

### Outliers

Foram identificados outliers pelo critério de 1,5 vezes o intervalo interquartil, concentrados em açúcar residual, cloretos e dióxido de enxofre total. Optou-se por mantê-los, por quatro razões:

1. São valores quimicamente plausíveis, não erros de medição.
2. O critério IQR pressupõe distribuição simétrica, e concentrações químicas são naturalmente assimétricas à direita. Parte do que é marcado como anomalia é artefato do método.
3. A base tem apenas 159 casos positivos. Remover linhas eliminaria exemplos raros justamente da classe que o modelo precisa aprender.
4. Random Forest é robusta a outliers por construção, já que divide por limiares.

## Pré-processamento

A divisão em 80% treino e 20% teste foi estratificada, preservando a proporção de classes nos dois conjuntos.

A padronização foi aplicada apenas aos modelos sensíveis à escala, sempre dentro de um Pipeline. Isso garante que média e desvio sejam calculados somente sobre o treino, prevenindo vazamento de dados.

A coluna `quality` foi removida das features. Como a variável binária é derivada dela, mantê-la causaria vazamento de alvo.

O parâmetro `class_weight="balanced"` foi usado para compensar o desbalanceamento, penalizando mais o erro na classe minoritária.

## Modelos

Foram treinados dois algoritmos com premissas diferentes.

A Regressão Logística serve como linha de base linear e interpretável. Vai encapsulada em Pipeline com StandardScaler, porque é sensível à escala das variáveis.

A Random Forest captura relações não lineares e interações entre variáveis. Não recebe padronização, já que cortes por limiar são invariantes à escala.

### Busca de hiperparâmetros

Foi executado um GridSearchCV com 108 combinações e 5 dobras, otimizando F1 em vez de acurácia e restrito ao conjunto de treino.

| Configuração            | F1 (CV treino)       | AUC (CV treino)      |
| ----------------------- | -------------------- | -------------------- |
| Random Forest padrão    | 0,621 (desvio 0,048) | 0,922 (desvio 0,017) |
| Random Forest otimizada | 0,641 (desvio 0,049) | 0,917 (desvio 0,022) |

A diferença de 0,020 em F1 é inferior a meio desvio padrão, ou seja, estatisticamente indistinguível de ruído. A busca confirmou que a configuração padrão já operava próximo do ótimo, comportamento esperado em Random Forest, que combate overfitting pela própria agregação de árvores descorrelacionadas.

A configuração otimizada foi adotada por ser mais regularizada, com profundidade máxima de 10 e mínimo de 4 amostras por folha. É uma decisão prudente em uma base com poucos exemplos positivos, e não um ganho de desempenho.

### Limiar de decisão

O limite padrão de 0,5 usado pelo método `predict` é uma convenção herdada de problemas balanceados. O limiar foi reotimizado para maximizar F1, usando probabilidades obtidas por validação cruzada sobre o treino, nunca sobre o teste.

O valor escolhido foi 0,61. Ele ficou acima de 0,5 porque o `class_weight="balanced"` já havia deslocado as probabilidades para cima durante o treino, e o limiar corrige esse excesso.

## Resultados

Avaliação sobre o conjunto de teste, com 229 amostras, das quais 32 são de alta qualidade.

| Modelo                      | Precisão | Recall | F1    | ROC-AUC | Acurácia |
| --------------------------- | -------- | ------ | ----- | ------- | -------- |
| Baseline trivial            | -        | 0,000  | 0,000 | 0,500   | 0,861    |
| Regressão Logística         | 0,379    | 0,688  | 0,489 | 0,850   | 0,799    |
| Random Forest (limiar 0,50) | 0,500    | 0,688  | 0,579 | 0,892   | 0,860    |
| Random Forest (limiar 0,61) | 0,579    | 0,688  | 0,629 | 0,892   | 0,886    |

As métricas de precisão, recall e F1 referem-se à classe de alta qualidade.

O modelo final atinge 88,6% de acurácia contra 86,1% do baseline trivial, um ganho aparente de apenas 2,5 pontos. O ROC-AUC de 0,892 revela o que a acurácia não mostra: dados um vinho bom e um comum escolhidos ao acaso, o modelo atribui probabilidade maior ao bom em 89% das vezes.

O ajuste do limiar reduziu falsos positivos sem custo. Ao elevar o corte de 0,50 para 0,61, os falsos positivos caíram de 22 para 16 sem perder nenhum vinho bom, já que o recall permaneceu em 0,688. Em termos operacionais, 6 lotes deixaram de ser classificados como premium indevidamente.

Os dois tipos de erro não custam o mesmo. Classificar um vinho comum como premium gera decepção do consumidor e dano à marca. Classificar um vinho bom como comum gera perda de margem, sem reclamação. Essa assimetria justifica priorizar a precisão.

A Random Forest supera a Regressão Logística em todas as métricas da classe positiva. A explicação está na natureza do problema. Com nenhuma variável isolada ultrapassando 0,41 de correlação, a qualidade depende de combinações condicionais que uma fronteira linear não consegue representar.

## Interpretação

| Variável        | Importância (RF) | Odds ratio | Leitura                                                             |
| --------------- | ---------------- | ---------- | ------------------------------------------------------------------- |
| Teor alcoólico  | 0,217            | 2,69       | Cada desvio padrão a mais quase triplica a chance de alta qualidade |
| Ácido cítrico   | 0,142            | 1,63       | Contribui com frescor e vivacidade                                  |
| Sulfatos        | 0,140            | 1,98       | Antioxidante, indica enologia bem conduzida                         |
| Acidez volátil  | 0,110            | 0,63       | Cada desvio padrão a mais reduz em 37% a chance                     |
| Açúcar residual | 0,046            | 1,39       | Praticamente irrelevante                                            |

As quatro primeiras variáveis respondem por 61% da importância total do modelo.

O teor alcoólico é um marcador, não uma causa. Álcool elevado resulta de uvas mais maduras, com maior concentração de açúcar na colheita, ou seja, sinaliza fruta de melhor qualidade. Adicionar álcool a um vinho ruim não o tornaria bom.

A acidez volátil tem relação causal direta. Trata-se essencialmente de ácido acético, um defeito de fermentação percebido como aroma de vinagre. É o único fator principal corrigível por controle de processo.

O açúcar residual é irrelevante para o modelo, o que contraria a associação popular entre doçura e qualidade.

Uma ressalva metodológica: a importância calculada por redução de impureza favorece variáveis contínuas com muitos valores distintos e dilui a importância entre variáveis correlacionadas. O topo do ranking é consistente entre os dois métodos, mas a ordenação exata das posições intermediárias deve ser lida com cautela.

## Implicações para a produção

O controle da acidez volátil é a alavanca de maior retorno. É o único dos quatro fatores principais sob controle direto do enólogo durante a vinificação, através de higiene de tanques, controle de exposição ao oxigênio e prevenção de contaminação por bactérias acéticas.

A definição do ponto de colheita pelo teor de açúcar da uva maximiza o potencial alcoólico, que é o preditor mais forte identificado.

A aplicação recomendada do modelo é a triagem automatizada de lotes, priorizando quais seguem para avaliação sensorial, e não a substituição do especialista. As 11 variáveis já são medidas rotineiramente em laboratório, o que torna o custo marginal de implementação praticamente nulo.

## Limitações

A base é pequena na classe de interesse. O conjunto de teste contém apenas 32 vinhos de alta qualidade, e cada acerto ou erro desloca o recall em cerca de 3 pontos percentuais.

O escopo está restrito a vinho tinto de uma única região produtora. A generalização para outros tipos ou origens não foi verificada.

A variável alvo é subjetiva. As notas refletem a percepção de avaliadores humanos, com variabilidade inerente. O modelo aprende a prever essa percepção, não uma medida objetiva de qualidade.

As relações identificadas são associativas, não causais. O modelo não autoriza inferências sobre o efeito de intervenções diretas nas variáveis.

## Próximos passos

Aplicação Streamlit para classificação interativa de novas amostras.

Teste de algoritmos de boosting, como XGBoost e LightGBM.

Extensão da análise a vinhos brancos, com tratamento diferenciado por tipo.

## Tecnologias

Python 3.11, pandas, NumPy, scikit-learn, Matplotlib, Seaborn e Jupyter.

## Autores

[Evanio Alves Carvalho · Michael Alves Pateis dos Santos · Guilherme Castilho Feitosa · Natália de Santana Cavalcanti]

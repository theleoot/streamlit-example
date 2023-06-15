import altair as alt
import math
import pandas as pd
import pycaret
import streamlit as st
from collections import namedtuple
from pycaret.regression import *

from load_data import load_data

"""
# Predizendo Albedo de Asteróides com IA

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

st.image("pictures/slide_1.png", caption="Slide 1")

columns_names = ["ASTEROID_NUMBER", "ASTEROID_NAME", "OCCULTATION_TIME",
                 "PHASE_ANGLE", "ECLIPTIC_LONGITUDE", "ECLIPTIC_LATITUDE",
                 "OCC_DIAM", "IRAS_DIAM", "IRAS_ALBEDO", "POLAR_ALBEDO",
                 "H", "OCC_ALB_FLAG", "OCC_ALBEDO", "FOUR_DEG_MAG"
                                                    "FOUR_DEG_ALBEDO", "QUAL_MAG", "QUAL_OCC"]

dataframe = pd.read_csv("data.csv", names=columns_names, delimiter=r"\s+", header=None, index_col=False)

"""
## Amostra dos Dados Utilizados

O conjunto de dados, a seguir, contém dados extraídos da pesquisa:

["Shevchenko, V.G. and Tedesco, E.F., Asteroid Albedos from Stellar Occultations V1.0. EAR-A-VARGBDET-5-OCCALB-V1.0. NASA Planetary Data System, 2007."](https://sbn.psi.edu/pds-staging/resource/occalb.html?refUrl=https%3A%2F%2Fsbn.psi.edu%2Fpds-staging%2Farchive%2Fasteroids.html&refName=Asteroid&type=Target+Type&typeUrl=https%3A%2F%2Fsbn.psi.edu%2Fpds%2Farchive%2Ftarget-types.html)


"""

st.table(dataframe.head())

st.image("pictures/slide_2.png", caption="Slide 2")

"""
## Módulos utilizados para criação do modelo

[PyCaret](https://pycaret.gitbook.io/docs/) é uma biblioteca Python que simplifica o processo de desenvolvimento de modelos de aprendizado de máquina. 
Ele fornece uma interface de alto nível, permitindo criar, comparar e avaliar rapidamente 
diferentes algoritmos de aprendizado de máquina.
"""

utilized_model_code = """import pycaret

from pycaret.regression import *"""

st.code(utilized_model_code, language="python")

"""
## Transformando os dados e criando o modelo.

A configuração do modelo de IA envolve coletar dados relevantes, pré-processá-los para 
remover ruídos e lidar com valores ausentes, selecionar uma arquitetura ou algoritmo de 
modelo apropriado, treinar o modelo usando os dados preparados e avaliar seu desempenho usando métricas 
específicas para a tarefa. O modelo treinado é então ajustado por meio de ajustes de parâmetros, validado
em dados não vistos e implantado em um ambiente de produção para fazer previsões sobre novos dados.
O monitoramento contínuo e a manutenção do modelo garantem sua precisão e confiabilidade ao longo do tempo.
"""

model_setup = """data_setup = setup(dataframe, target="H", session_id=123)"""

st.code(model_setup, language="python")

st.image("pictures/model_info.png", caption="Informações do Modelo Criado")

"""
## Selecionando o modelo mais adequado

A seleção do modelo de IA é o processo de escolha do algoritmo ou arquitetura mais adequada para uma tarefa 
específica. Envolve considerar fatores como o tipo de problema (classificação, regressão, agrupamento), características
dos dados e métricas de desempenho desejadas. A seleção de modelos visa encontrar a melhor correspondência entre o problema
em questão e as capacidades de diferentes algoritmos. Pode envolver a comparação e avaliação de vários modelos, considerando
seus pontos fortes e fracos e selecionando aquele com maior probabilidade de obter previsões precisas e confiáveis. O modelo
escolhido se torna a base para treinamento, avaliação e implantação do sistema de IA.
"""

model_comparison = """models_comparison = compare_models()"""

st.code(model_comparison, language="python")

st.image("pictures/model_comparison.png", caption="Comparação entre os modelos criados")

code = """data_setup = setup(dataframe, target="H", session_id=123)
models_comparison = compare_models()"""

"""
## Análise do erro gerado pelo modelo

A análise de erro do modelo IA envolve o exame dos erros ou imprecisões cometidos pelo modelo 
durante as previsões. Ele ajuda a identificar padrões, entender os tipos de erros cometidos e
revelar insights no qual modelo pode estar com problemas. Essa análise fornece 
feedback valioso para melhoria e refinamento do modelo.
"""

st.image("pictures/residuals.png", caption="Análise dos resíduos gerados pelo modelo")
st.image("pictures/error.png", caption="Análise dos erros gerados pelo modelo")

"""
## Selecionando amostra dos dados para teste

Uma amostra de teste de modelo IA refere-se a um subconjunto de dados que é retirado do 
processo de treinamento e usado exclusivamente para avaliar o desempenho do modelo. Essas 
amostras de teste representam dados do mundo real que o modelo não viu antes, permitindo-nos
avaliar o quão bem o modelo generaliza para instâncias novas e não vistas. O desempenho na 
amostra de teste fornece informações sobre a precisão e eficácia do modelo em cenários do mundo real.
"""

with st.echo(code_location='below'):
    sample_data = dataframe.drop(columns=["H"], axis=1).head()

    sample_data = sample_data.assign(ASTEROID_NUMBER=3200)
    sample_data = sample_data.assign(ASTEROID_NAME="sample")

    st.table(sample_data.head())

"""
## Importando o modelo treinado
"""
with st.echo(code_location='below'):
    model = load_model("test_pipeline")

"""
## Predizendo os dados de teste

A previsão do modelo IA refere-se ao processo de usar um modelo treinado
para fazer previsões ou gerar saídas para dados novos ou não vistos. O modelo
recebe os dados de entrada e produz uma previsão ou inferência com base nos padrões
e relacionamentos aprendidos na fase de treinamento. Isso permite que o modelo forneça
recursos inteligentes e automatizados de tomada de decisão ou previsão.
"""

with st.echo(code_location='below'):
    predictions = predict_model(model, sample_data)
    st.table(predictions.head())

"""
## Teste o modelo...
"""

custom_data = dataframe.head(1)
edited_df = st.data_editor(custom_data.loc[:, custom_data.columns != 'H'])

if st.button("Predizer"):
    st.table(predict_model(model, edited_df))

st.image("pictures/slide_6.png", caption="Slide 6")

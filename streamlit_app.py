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
## Exemplo de Valores no Database.
"""

st.table(dataframe.head())

st.image("pictures/slide_2.png", caption="Slide 2")

"""
## Módulos utilizados para criação do modelo
"""

utilized_model_code = """import pycaret

from pycaret.regression import *"""

st.code(utilized_model_code, language="python")

"""
## Transformando os dados e criando o modelo.
"""

model_setup = """data_setup = setup(dataframe, target="H", session_id=123)"""

st.code(model_setup, language="python")

st.image("pictures/model_info.png", caption="Informações do Modelo Criado")

"""
## Selecionando o modelo mais adequado.
"""

model_comparison = """models_comparison = compare_models()"""

st.code(model_comparison, language="python")

st.image("pictures/model_comparison.png", caption="Comparação entre os modelos criados")

code = """data_setup = setup(dataframe, target="H", session_id=123)
models_comparison = compare_models()"""

"""
## Análise do erro gerado pelo modelo.
"""

st.image("pictures/residuals.png", caption="Análise dos resíduos gerados pelo modelo")
st.image("pictures/error.png", caption="Análise dos erros gerados pelo modelo")


sample_data = dataframe.drop(columns=["H"], axis=1).head()

sample_data = sample_data.assign(ASTEROID_NUMBER=3200)
sample_data = sample_data.assign(ASTEROID_NAME="sample")

st.table(sample_data.head())

model = load_model("test_pipeline")

predictions = predict_model(model, sample_data)

st.table(predictions.head())

st.image("pictures/slide_6.png", caption="Slide 6")
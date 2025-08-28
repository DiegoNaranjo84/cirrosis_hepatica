# Cargue de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import kagglehub
import os
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Cirrosis Hepatica Streamlit App", layout="wide")
st.title("Clasificación de los estadios de la cirrosis hepática con métodos de Machine Learning")

st.caption("Estudio clínico de cirrosis hepática — ficha de variables")

texto = """
### **Variables:**

* **N_Days**: Número de días transcurridos entre el registro y la fecha más temprana entre fallecimiento, trasplante o análisis del estudio en 1986.  
* **Status**: estado del paciente C (censurado), CL (censurado por tratamiento hepático) o D (fallecimiento).  
* **Drug**: tipo de fármaco: D-penicilamina o placebo.  
* **Age**: edad en días.  
* **Sex**: M (hombre) o F (mujer).  
* **Ascites**: presencia de ascitis N (No) o Y (Sí).  
* **Hepatomegaly**: presencia de hepatomegalia N (No) o Y (Sí).  
* **Spiders**: presencia de aracnosis N (No) o Y (Sí).  
* **Edema**: presencia de edema N (sin edema ni tratamiento diurético), S (edema presente sin diuréticos o resuelto con diuréticos) o Y (edema a pesar del tratamiento diurético).  
* **Bilirubin**: bilirrubina sérica en mg/dl.  
* **Cholesterol**: colesterol sérico en mg/dl.  
* **Albumin**: albúmina en g/dl.  
* **Copper**: cobre en orina en µg/día.  
* **Alk_Phos**: fosfatasa alcalina en U/litro.  
* **SGOT**: SGOT en U/ml.  
* **Tryglicerides**: triglicéridos en mg/dl.  
* **Platelets**: plaquetas por metro cúbico [ml/1000].  
* **Prothrombin**: tiempo de protrombina en segundos.  
* **Stage**: estadio histológico de la enfermedad (1, 2 o 3).  

---

### **Dimensiones del dataset**
- **Tamaño:** 25 000 filas, 19 columnas  
- **Faltantes:** 0% en todas las columnas  

---
"""

st.markdown(texto)

# --- Cargar dataset ---
# URL del CSV en GitHub (raw)
url = "https://raw.githubusercontent.com/DiegoNaranjo84/cirrosis_hepatica/main/liver_cirrhosis.csv"

# Cargar el dataset
df = pd.read_csv(url)

# Filtrar solo columnas categóricas (tipo "object" o "category")
cat_cols = df.select_dtypes(include=['object', 'category'])

st.subheader("Primeras 10 filas del dataset")
st.dataframe(df.head(10), use_container_width=True)

# ------- Helpers -------
def format_uniques(series, max_items=20):
    """Convierte valores únicos a una cadena legible, acota a max_items."""
    uniques = pd.Series(series.dropna().unique())
    head = uniques.head(max_items).astype(str).tolist()
    txt = ", ".join(head)
    if uniques.size > max_items:
        txt += f" … (+{uniques.size - max_items} más)"
    return txt

# ------- Detectar tipos -------
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

# ------- Resumen variables categóricas -------
cat_summary = pd.DataFrame({
    "Variable": cat_cols,
    "Tipo de dato": [df[c].dtype for c in cat_cols],
    "Nº de categorías únicas": [df[c].nunique(dropna=True) for c in cat_cols],
    "Nº de datos no nulos": [df[c].notna().sum() for c in cat_cols],
    "Categorías": [format_uniques(df[c], max_items=20) for c in cat_cols],
})

# ------- Resumen variables numéricas -------
num_summary = pd.DataFrame({
    "Variable": num_cols,
    "Tipo de dato": [df[c].dtype for c in num_cols],
    "Nº de datos no nulos": [df[c].notna().sum() for c in num_cols],
    "Mínimo": [df[c].min(skipna=True) for c in num_cols],
    "Máximo": [df[c].max(skipna=True) for c in num_cols],
    "Media":  [df[c].mean(skipna=True) for c in num_cols],
    "Desviación estándar": [df[c].std(skipna=True) for c in num_cols],
}).round(2)

# ------- Mostrar en dos columnas iguales con separación uniforme -------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Resumen variables categóricas")
    st.dataframe(cat_summary, use_container_width=True)

with col2:
    st.subheader("Resumen variables numéricas")
    st.dataframe(num_summary, use_container_width=True)

#####--------------------------------------------------------------------------------------#########

st.markdown("""### Análisis de variables categóricas""")
st.caption("Selecciona una variable para ver su distribución en tabla y gráfico de torta.")

variables_categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

if not variables_categoricas:
    st.warning("No se detectaron variables categóricas (object/category/bool) en df.")
    st.stop()

# =========================
# Controles (En la sección)
# =========================
st.markdown("*Controles*")
with st.container():
    c1, c2 = st.columns([1.6, 1.1])
    with c1:
        var = st.selectbox(
            "Variable categórica",
            options=variables_categoricas,
            index=0,
            key="cat_var_local"
        )
        top_n = st.slider(
            "Top N (agrupa el resto en 'Otros')",
            min_value=3, max_value=30, value=10, step=1,
            help="Agrupa las categorías menos frecuentes en 'Otros'",
            key="cat_topn_local"
        )
    with c2:
        orden_alfabetico = st.checkbox("Ordenar alfabéticamente (solo tabla)", value=False, key="cat_orden_local")

# =========================
# Preparar datos (siempre incluye NaN)
# =========================
serie = df[var].copy()

vc = serie.value_counts(dropna=True)  # Se mantienen los NaN para contarlos

# Etiqueta amigable para NaN
labels = vc.index.to_list()
labels = ["(NaN)" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in labels]
counts = vc.values

data = pd.DataFrame({"Categoría": labels, "Conteo": counts})
data["Porcentaje"] = (data["Conteo"] / data["Conteo"].sum() * 100).round(2)

# Agrupar en "Otros" si supera Top N
if len(data) > top_n:
    top = data.nlargest(top_n, "Conteo").copy()
    otros = data.drop(top.index)
    fila_otros = pd.DataFrame({
        "Categoría": ["Otros"],
        "Conteo": [int(otros["Conteo"].sum())],
        "Porcentaje": [round(float(otros["Porcentaje"].sum()), 2)]
    })
    data_plot = pd.concat([top, fila_otros], ignore_index=True)
else:
    data_plot = data.copy()

# Orden por Conteo (siempre)
data_plot = data_plot.sort_values("Conteo", ascending=False).reset_index(drop=True)

# Orden opcional alfabético en la tabla (no afecta el gráfico)
data_table = data_plot.copy()
if orden_alfabetico:
    data_table = data_table.sort_values("Categoría").reset_index(drop=True)

# =========================
# Mostrar tabla y gráfico
# =========================
tcol, gcol = st.columns([1.1, 1.3], gap="large")

with tcol:
    st.subheader(f"Distribución de {var}")
    st.dataframe(
        data_table.assign(Porcentaje=data_table["Porcentaje"].round(2)),
        use_container_width=True
    )

with gcol:
    st.subheader("Gráfico de torta")
    chart = (
        alt.Chart(data_plot)
        .mark_arc(outerRadius=110)
        .encode(
            theta=alt.Theta(field="Conteo", type="quantitative"),
            color=alt.Color("Categoría:N", legend=alt.Legend(title="Categoría")),
            tooltip=[
                alt.Tooltip("Categoría:N"),
                alt.Tooltip("Conteo:Q", format=","),
                alt.Tooltip("Porcentaje:Q", format=".2f")
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# Extras informativos
# =========================
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.metric("Categorías mostradas", f"{len(data_plot)}")
with c2:
    st.metric("Total registros (variable seleccionada)", f"{int(serie.shape[0]):,}".replace(",", "."))

st.caption("Consejo: usa *Top N* para simplificar la lectura y agrupar categorías poco frecuentes en 'Otros'.")

#####--------------------------------------------------------------------------------------#########

# =========================
# Análisis de variables numéricas
# =========================
st.markdown("""### Análisis de variables numéricas""")
st.caption("Selecciona una variable para ver su distribución en tabla, boxplot e histograma.")

# Detectar variables numéricas
variables_numericas = df.select_dtypes(include=["number"]).columns.tolist()

if not variables_numericas:
    st.warning("No se detectaron variables numéricas en df.")
    st.stop()

# =========================
# Controles dentro de la sección
# =========================
st.markdown("*Controles*")
with st.container():
    c1, c2 = st.columns([1.6, 1.4])
    with c1:
        var_num = st.selectbox(
            "Variable numérica",
            options=variables_numericas,
            index=0,
            key="num_var_local"
        )
    with c2:
        bins = st.slider(
            "Número de bins (histograma)",
            min_value=5, max_value=100, value=30, step=5,
            key="num_bins_local"
        )

# =========================
# Preparar serie
# =========================
serie_num = df[var_num].dropna()

# =========================
# Métricas descriptivas
# =========================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Nº datos no nulos", f"{serie_num.shape[0]:,}".replace(",", "."))
with c2:
    st.metric("Mínimo", f"{serie_num.min():.2f}")
with c3:
    st.metric("Máximo", f"{serie_num.max():.2f}")
with c4:
    st.metric("Media", f"{serie_num.mean():.2f}")
with c5:
    st.metric("Desv. Estándar", f"{serie_num.std():.2f}")

# =========================
# Gráficos
# =========================
g1, g2 = st.columns(2, gap="large")

# --- Boxplot vertical y ancho ---
with g1:
    st.subheader(f"Boxplot de {var_num} (vertical)")
    box_data = pd.DataFrame({var_num: serie_num})
    box_data["_grupo_"] = "Distribución"  # ancla un grupo único en X

    box_chart = (
        alt.Chart(box_data)
        .mark_boxplot(size=140, extent=1.5)  # size = ancho de la caja; extent=1.5 => whiskers tipo Tukey
        .encode(
            x=alt.X("_grupo_:N", axis=None, title=""),
            y=alt.Y(f"{var_num}:Q", title=var_num)
        )
        .properties(height=350)
    )
    st.altair_chart(box_chart, use_container_width=True)

# --- Histograma ---
with g2:
    st.subheader(f"Histograma de {var_num}")
    hist_data = pd.DataFrame({var_num: serie_num})
    hist_chart = (
        alt.Chart(hist_data)
        .mark_bar()
        .encode(
            alt.X(var_num, bin=alt.Bin(maxbins=bins)),
            y='count()',
            tooltip=[
                alt.Tooltip(var_num, bin=alt.Bin(maxbins=bins)),
                alt.Tooltip('count()', title="Frecuencia")
            ]
        )
        .properties(height=350)
    )
    st.altair_chart(hist_chart, use_container_width=True)

# =========================
# Matriz de Correlación
# =========================
st.markdown("### Matriz de Correlación")
correlacion = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Matriz de Correlación")
st.pyplot(fig)

# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 1. Selección de carácteristicas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.1. Selección de carácteristicas categóricas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.2. Selección de carácteristicas numéricas""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 1.3. Unión de variables categóricas y númericas""")



# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 2. MCA Y PCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.1. MCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""## 2.2. PCA""")
# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 3. RFE""")

# ______________________________________________________

st.markdown("""# 3. RFE""")

# Convertir Stage a categórica
df["Stage"] = df["Stage"].astype("category")

# Definir variables categóricas y numéricas
categorical = df.select_dtypes(include=["object","category"])
categorical_features = categorical.columns.drop("Stage").tolist()
numerical_features = df.select_dtypes(include=["int64","float64"]).columns.tolist()

# Separar X e y
X = df[categorical_features + numerical_features]
y = df["Stage"]

# Partición train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# OneHotEncoder compatible con distintas versiones de sklearn
try:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)

# Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OHE, categorical_features),
    ]
)

# Modelos disponibles
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

# ---- Control: seleccionar 1 modelo (por defecto Random Forest) ----
model_names = list(models.keys())
default_index = model_names.index("Decision Tree") if "Random Forest" in model_names else 0
modelo_elegido = st.selectbox("Modelo a ejecutar", options=model_names, index=default_index, key="rfe_modelo")
model = models[modelo_elegido]

st.title("Resultados de Selección de Características con RFE-CV")
st.subheader(f"Modelo: {modelo_elegido}")

# RFECV
rfe = RFECV(
    estimator=model,
    step=1,
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),
    scoring="accuracy",
    n_jobs=-1,
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selection", rfe),
    ("model", model),
])

# Entrenar
pipeline.fit(X_train, y_train)

# Evaluar
accuracy_test = pipeline.score(X_test, y_test)
mask = pipeline.named_steps["feature_selection"].support_
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
selected_names = feature_names[mask]

# Mostrar en la app
st.write(f"*Accuracy en test set:* {accuracy_test:.3f}")
st.write(f"*Variables seleccionadas:* {len(selected_names)}")
st.write(f"*Nombres:* {list(selected_names)}")

# Resumen final (solo el modelo elegido)
st.header("Resumen Final")
st.markdown(f"""
*Modelo:* {modelo_elegido}  
- Accuracy: {accuracy_test:.3f}  
- Variables seleccionadas: {len(selected_names)}  
- Nombres: {list(selected_names)}  
""")

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


# Descargar el dataset
path = kagglehub.dataset_download("aadarshvelu/liver-cirrhosis-stage-classification")
print("Ruta local del dataset:", path)

# Ver los archivos del dataset cargado
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

file_path = os.path.join(path, "liver_cirrhosis.csv")
df = pd.read_csv(file_path)

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
    st.warning("No se detectaron variables categóricas (object/category/bool) en `df`.")
    st.stop()


# =========================
# Controles (Sidebar)
# =========================
st.sidebar.header("Controles")
var = st.sidebar.selectbox("Variable categórica", options=variables_categoricas, index=0)
incluir_na = st.sidebar.checkbox("Incluir NaN", value=True)
metric_opt = st.sidebar.radio("Métrica", options=["Porcentaje", "Conteo"], index=0)
top_n = st.sidebar.slider("Top N", min_value=3, max_value=30, value=10, step=1, help="Agrupa las categorías menos frecuentes en 'Otros'")
orden_alfabetico = st.sidebar.checkbox("Ordenar categorías alfabéticamente (en la tabla)", value=False)

# =========================
# Preparar datos
# =========================
serie = df[var].copy()

if not incluir_na:
    serie = serie.dropna()

vc = serie.value_counts(dropna=incluir_na)

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

# Orden por métrica elegida para el gráfico
metric = "Porcentaje" if metric_opt == "Porcentaje" else "Conteo"
data_plot = data_plot.sort_values(metric, ascending=False).reset_index(drop=True)

# Orden opcional alfabético en la tabla (no afecta el gráfico)
data_table = data_plot.copy()
if orden_alfabetico:
    data_table = data_table.sort_values("Categoría").reset_index(drop=True)

# =========================
# Mostrar tabla y gráfico
# =========================
tcol, gcol = st.columns([1.1, 1.3], gap="large")

with tcol:
    st.subheader(f"Distribución de `{var}`")
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
            theta=alt.Theta(field=metric, type="quantitative"),
            color=alt.Color("Categoría:N", legend=alt.Legend(title="Categoría")),
            tooltip=[
                alt.Tooltip("Categoría:N"),
                alt.Tooltip("Conteo:Q", format=","),
                alt.Tooltip("Porcentaje:Q", format=".2f")
            ],
        )
        .properties(width="container", height=380)
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# Extras informativos
# =========================
st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Categorías mostradas", f"{len(data_plot)}")
with c2:
    st.metric("Total registros (variable seleccionada)", f"{int(serie.shape[0]):,}".replace(",", "."))
with c3:
    st.metric("Incluye NaN", "Sí" if incluir_na else "No")

st.caption("Consejo: usa **Top N** para simplificar la lectura y agrupar categorías poco frecuentes en 'Otros'.")





#####--------------------------------------------------------------------------------------#########


# =========================
# Análisis de variables numéricas
# =========================
st.markdown("""### Análisis de variables numéricas""")
st.caption("Selecciona una variable para ver su distribución en tabla, boxplot e histograma.")

# Detectar variables numéricas
variables_numericas = df.select_dtypes(include=["number"]).columns.tolist()

if not variables_numericas:
    st.warning("No se detectaron variables numéricas en `df`.")
    st.stop()

# Controles (Sidebar)
st.sidebar.header("Controles - Numéricas")
var_num = st.sidebar.selectbox("Variable numérica", options=variables_numericas, index=0, key="num_var")
bins = st.sidebar.slider("Número de bins (histograma)", min_value=5, max_value=100, value=30, step=5)

# Preparar serie
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

with g1:
    st.subheader(f"Boxplot de `{var_num}`")
    box_data = pd.DataFrame({var_num: serie_num})
    box_chart = (
        alt.Chart(box_data)
        .mark_boxplot()
        .encode(y=alt.Y(var_num, type="quantitative"))
        .properties(height=300)
    )
    st.altair_chart(box_chart, use_container_width=True)

with g2:
    st.subheader(f"Histograma de `{var_num}`")
    hist_data = pd.DataFrame({var_num: serie_num})
    hist_chart = (
        alt.Chart(hist_data)
        .mark_bar()
        .encode(
            alt.X(var_num, bin=alt.Bin(maxbins=bins)),
            y='count()',
            tooltip=[alt.Tooltip(var_num, bin=alt.Bin(maxbins=bins)), alt.Tooltip('count()', title="Frecuencia")]
        )
        .properties(height=300)
    )
    st.altair_chart(hist_chart, use_container_width=True)


st.markdown("### Matriz de Correlación")

correlacion = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Matriz de Correlación")

st.pyplot(fig) 


# ________________________________________________________________________________________________________________________________________________________________
st.markdown("""# 1. Selección de carácteristicas""")
st.markdown("""# 1. Ensayo""")

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
# ________________________________________________________________________________________________________________________________________________________________


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jflorez-giraldo/Nhanes-streamlit/main/nhanes_2015_2016.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Asignar condiciones
def assign_condition(row):
    if (row["BPXSY1"] >= 140 or row["BPXSY2"] >= 140 or 
        row["BPXDI1"] >= 90 or row["BPXDI2"] >= 90):
        return "hypertension"
    elif row["BMXBMI"] >= 30:
        return "diabetes"
    elif ((row["RIAGENDR"] == 1 and row["BMXWAIST"] > 102) or 
          (row["RIAGENDR"] == 2 and row["BMXWAIST"] > 88)):
        return "high cholesterol"
    else:
        return "healthy"

df["Condition"] = df.apply(assign_condition, axis=1)

# Diccionario de códigos por variable categórica
category_mappings = {
    "RIAGENDR": {
        1: "Male",
        2: "Female"
    },
    "DMDMARTL": {
        1: "Married",
        2: "Divorced",
        3: "Never married",
        4: "Widowed",
        5: "Separated",
        6: "Living with partner",
        77: "Refused",
        99: "Don't know"
    },
    "DMDEDUC2": {
        1: "Less than 9th grade",
        2: "9-11th grade (no diploma)",
        3: "High school/GED",
        4: "Some college or AA degree",
        5: "College graduate or above",
        7: "Refused",
        9: "Don't know"
    },
    "SMQ020": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "ALQ101": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "ALQ110": {
        1: "Every day",
        2: "5–6 days/week",
        3: "3–4 days/week",
        4: "1–2 days/week",
        5: "2–3 days/month",
        6: "Once a month or less",
        7: "Refused",
        9: "Don't know"
    },
    "RIDRETH1": {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        5: "Other Race - Including Multi-Racial"
    },
    "DMDCITZN": {
        1: "Citizen by birth or naturalization",
        2: "Not a citizen of the U.S.",
        7: "Refused",
        9: "Don't know"
    },
    "HIQ210": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "SDMVPSU": {
        1: "PSU 1",
        2: "PSU 2"
    },
    "DMDHHSIZ": {
        1: "1 person",
        2: "2 people",
        3: "3 people",
        4: "4 people",
        5: "5 people",
        6: "6 people",
        7: "7 or more people"
    }
}

def apply_categorical_mappings(df, mappings):
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

df = apply_categorical_mappings(df, category_mappings)

col_map = {
    "SEQN": "Participant ID",
    "ALQ101": "Alcohol Intake - Past 12 months (Q1)",
    "ALQ110": "Alcohol Frequency",
    "ALQ130": "Alcohol Amount",
    "SMQ020": "Smoking Status",
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age (years)",
    "RIDRETH1": "Race/Ethnicity",
    "DMDCITZN": "Citizenship",
    "DMDEDUC2": "Education Level",
    "DMDMARTL": "Marital Status",
    "DMDHHSIZ": "Household Size",
    "WTINT2YR": "Interview Weight",
    "SDMVPSU": "Masked PSU",
    "SDMVSTRA": "Masked Stratum",
    "INDFMPIR": "Income to Poverty Ratio",
    "BPXSY1": "Systolic BP1",
    "BPXDI1": "Diastolic BP1",
    "BPXSY2": "Systolic BP2",
    "BPXDI2": "Diastolic BP2",
    "BMXWT": "Body Weight",
    "BMXHT": "Body Height",
    "BMXBMI": "Body Mass Index",
    "BMXLEG": "Leg Length",
    "BMXARML": "Arm Length",
    "BMXARMC": "Arm Circumference",
    "BMXWAIST": "Waist Circumference",
    "HIQ210": "Health Insurance Coverage"
}

df = df.rename(columns=col_map)

# Asegurar compatibilidad con Arrow/Streamlit
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("string").fillna("Missing")

df = df.reset_index(drop=True)

# Mostrar info y variables categóricas lado a lado
st.subheader("Resumen de Datos")

# Crear columnas para mostrar info_df y category_df lado a lado
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Tipo de Dato y Nulos**")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Dtype": df.dtypes.values
    })
    st.dataframe(info_df, use_container_width=True)

# Detección automática de variables categóricas
categorical_vars = [col for col in df.columns 
                    if df[col].dtype == 'object' or 
                       df[col].dtype == 'string' or 
                       df[col].nunique() <= 10]

with col2:
    st.markdown("**Variables Categóricas Detectadas**")
    category_info = []
    for col in categorical_vars:
        unique_vals = df[col].dropna().unique()
        category_info.append({
            "Variable": col,
            "Unique Classes": ", ".join(map(str, sorted(unique_vals)))
        })

    category_df = pd.DataFrame(category_info)
    st.dataframe(category_df, use_container_width=True)


st.subheader("Primeras 10 filas del dataset")
st.dataframe(df.head(10), use_container_width=True)

# Filtros
with st.sidebar:
    st.header("Filters")
    gender_filter = st.multiselect("Gender", sorted(df["Gender"].dropna().unique()))
    race_filter = st.multiselect("Race/Ethnicity", sorted(df["Race/Ethnicity"].dropna().unique()))
    condition_filter = st.multiselect("Condition", sorted(df["Condition"].dropna().unique()))
    #st.markdown("---")
    #k_vars = st.slider("Number of variables to select", 2, 10, 5)

# Aplicar filtros
for col, values in {
    "Gender": gender_filter, "Race/Ethnicity": race_filter, "Condition": condition_filter
}.items():
    if values:
        df = df[df[col].isin(values)]

if df.empty:
    st.warning("No data available after applying filters.")
    st.stop()

# Mostrar advertencias
problematic_cols = df.columns[df.dtypes == "object"].tolist()
nullable_ints = df.columns[df.dtypes.astype(str).str.contains("Int64")].tolist()

st.write("### ⚠️ Columnas potencialmente problemáticas para Arrow/Streamlit:")
if problematic_cols or nullable_ints:
    st.write("**Tipo 'object':**", problematic_cols)
    st.write("**Tipo 'Int64' (nullable):**", nullable_ints)
else:
    st.success("✅ No hay columnas problemáticas detectadas.")


# ==============================
# TRAIN / TEST SPLIT ANTES DE PCA y MCA
# ==============================
# 1️⃣ Eliminar filas con NaN en Condition
df = df.dropna(subset=["Condition"])

# --- 2. Separar X (features) y y (target) ---
X_df = df.drop(columns=["Condition"])
y_df = df["Condition"]

# --- 3. Identificar variables numéricas y categóricas ---
numeric_features = X_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# --- 4. Pipelines de imputación y escalado ---
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# --- 5. Preprocesador general ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# --- 6. Separar datos en train y test antes de PCA/MCA ---
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_df, test_size=0.2, random_state=42, stratify=y_df
)

# Mostrar info de división
st.write(f"Datos de entrenamiento: {X_train.shape[0]} filas")
st.write(f"Datos de prueba: {X_test.shape[0]} filas")

# --- 7. Transformar datos (sin PCA/MCA todavía) ---
# Ajustar el preprocesador
preprocessor.fit(X_train)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Obtener datos numéricos procesados
num_data_train = preprocessor.named_transformers_['num'].transform(X_train[numeric_features])
num_data_test = preprocessor.named_transformers_['num'].transform(X_test[numeric_features])

# Obtener datos categóricos procesados
cat_data_train = preprocessor.named_transformers_['cat'].transform(X_train[categorical_features])
cat_data_test = preprocessor.named_transformers_['cat'].transform(X_test[categorical_features])

# PCA sobre numéricas
from sklearn.decomposition import PCA
pca = PCA(n_components=7)
X_train_pca = pca.fit_transform(num_data_train)
X_test_pca = pca.transform(num_data_test)

pca = PCA(n_components=2)  # si quieres todas, usa n_components=None
X_pca = pca.fit_transform(X_train_processed)

# Crear DataFrame con resultados
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['condition'] = y_train.values  # color por la variable objetivo

# --- 1. Scatterplot PC1 vs PC2 ---
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='condition', palette='viridis', alpha=0.7)
plt.title('PCA - PC1 vs PC2')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.legend(title='Condition')
plt.tight_layout()
plt.show()

## Obtener los loadings del PCA (componentes * características)
#loadings = X_pca.named_steps["pca"].components_

## Convertir a DataFrame con nombres de columnas
#loadings_df = pd.DataFrame(
#    loadings,
#    columns=X_num.columns,
#    index=[f"PC{i+1}" for i in range(loadings.shape[0])]
#).T  # Transponer para que columnas sean PCs y filas las variables

## Ordenar las filas por la importancia de la variable en la suma de cuadrados de los componentes
## Esto agrupa por aquellas variables con mayor contribución total
#loading_magnitude = (loadings_df**2).sum(axis=1)
#loadings_df["Importance"] = loading_magnitude
#loadings_df_sorted = loadings_df.sort_values(by="Importance", ascending=False).drop(columns="Importance")

## Graficar heatmap ordenado
#st.subheader("🔍 Heatmap de Loadings del PCA (Componentes Principales)")

#fig, ax = plt.subplots(figsize=(10, 12))
#sns.heatmap(loadings_df_sorted, annot=True, cmap="coolwarm", center=0, ax=ax)
#st.pyplot(fig)

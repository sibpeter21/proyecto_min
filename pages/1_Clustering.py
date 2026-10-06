import json
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Modelo 1 · Clustering", layout="wide")


@st.cache_resource
def cargar_modelo():
    return pickle.load(open("modelos/modelo_clustering.pkl", "rb"))


kmeans, preprocesamiento, variables = cargar_modelo()
info = json.load(open("modelos/info_clustering.json", encoding="utf-8"))
perfiles = pd.read_csv("datos/perfiles_clusters.csv")

# Categorías válidas, tomadas del propio preprocesamiento entrenado
cols_categoricas = preprocesamiento.transformers_[1][2]
categorias = dict(zip(cols_categoricas, preprocesamiento.named_transformers_["categoricas"].categories_))

st.title("Modelo 1 · Perfil de un trabajador (clustering)")
st.write(
    "K-Means agrupó a los trabajadores del centro de control en cinco perfiles según sus características "
    "sociodemográficas y laborales. Ingresa los datos de un trabajador para ver a qué perfil se parece más."
)

col_in, col_out = st.columns([1, 1], gap="large")
r = info["rangos"]

with col_in:
    st.subheader("Datos del trabajador")
    edad = st.slider("Edad", r["Edad"][0], r["Edad"][1], (r["Edad"][0] + r["Edad"][1]) // 2)
    hijos = st.slider("Número de hijos", r["Hijos"][0], r["Hijos"][1], r["Hijos"][0])
    antiguedad = st.slider("Antigüedad (años)", r["Antigüedad"][0], r["Antigüedad"][1],
                           (r["Antigüedad"][0] + r["Antigüedad"][1]) // 2)
    estado_civil = st.selectbox("Estado civil", list(categorias["Estado Civil"]))
    casa = st.selectbox("Vivienda", list(categorias["Casa Propia"]))
    transporte = st.selectbox("Medio de transporte", list(categorias["Medio Transporte"]))
    cargo = st.selectbox("Cargo", list(categorias["Cargo"]))
    equipo = st.selectbox("Equipo de trabajo", list(categorias["Homologacion Equipo de trabajo"]))

# Dataframe con los mismos nombres y orden de variables del notebook
valores = {"Edad": edad, "Estado Civil": estado_civil, "Hijos": hijos, "Casa Propia": casa,
           "Medio Transporte": transporte, "Antigüedad": antiguedad, "Cargo": cargo,
           "Homologacion Equipo de trabajo": equipo}
nuevo = pd.DataFrame([[valores[v] for v in variables]], columns=variables)
cluster = int(kmeans.predict(preprocesamiento.transform(nuevo))[0])
perfil = info["perfiles"][str(cluster)]
fila = perfiles[perfiles["Cluster"] == cluster].iloc[0]

with col_out:
    st.subheader("Resultado")
    st.success(f"Perfil asignado: **{perfil}**  (cluster {cluster})")
    a, b, c = st.columns(3)
    a.metric("Edad promedio del perfil", f"{fila['Edad promedio']:.0f} años")
    b.metric("Hijos promedio", f"{fila['Hijos promedio']:.1f}")
    c.metric("Antigüedad promedio", f"{fila['Antigüedad promedio']:.0f} años")
    st.write(
        f"Este perfil reúne a **{int(fila['Número de trabajadores'])} trabajadores**. "
        f"Predomina el cargo **{fila['Cargo predominante']}** ({fila['% Cargo']:.0f} %) "
        f"y el equipo **{fila['Homologacion Equipo de trabajo predominante']}** "
        f"({fila['% Homologacion Equipo de trabajo']:.0f} %)."
    )

st.subheader("Los cinco perfiles")
tabla = perfiles[["Cluster", "Perfil", "Número de trabajadores", "Edad promedio", "Hijos promedio",
                  "Antigüedad promedio", "Cargo predominante", "Homologacion Equipo de trabajo predominante"]].rename(
    columns={"Homologacion Equipo de trabajo predominante": "Equipo predominante"})
st.dataframe(tabla, hide_index=True, width="stretch")

with st.expander("Calidad de la agrupación y limitaciones"):
    a, b = st.columns(2)
    a.metric("Silhouette", f"{info['silhouette']:.2f}")
    b.metric("Davies-Bouldin", f"{info['davies_bouldin']:.2f}")
    st.write(
        "La separación entre grupos es limitada (Silhouette bajo): los perfiles son tendencias de la población "
        "y no grupos completamente separados. Los rangos de edad, hijos y antigüedad son los de la población caracterizada."
    )

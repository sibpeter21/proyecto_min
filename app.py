import pandas as pd
import streamlit as st

st.set_page_config(page_title="Minería de datos · Centro de control", page_icon="📊", layout="wide")

st.title("Proyecto de minería de datos · Centro de control de energía")
st.write(
    "Aplicación del grupo con **tres modelos** construidos sobre datos de llamadas de Teams y de la "
    "caracterización del personal de un centro de control de energía. Usa el menú de la izquierda para abrir cada modelo. "
    "Esta página es solo **informativa**: muestra el contexto de cada **equipo de trabajo**, que es la variable que conecta "
    "las dos bases de datos."
)

# ---------------------------------------------------------------- Modelos
st.subheader("Modelos del proyecto")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**1 · Clustering de trabajadores**")
    st.caption("Asigna un trabajador a uno de cinco perfiles según sus características sociodemográficas y laborales.")
    st.page_link("pages/1_Clustering.py", label="Abrir modelo 1")
with c2:
    st.markdown("**2 · Regresión: duración de llamadas**")
    st.caption("Estima la duración esperada de una llamada atendida según equipo, hora, día y dirección.")
    st.page_link("pages/2_Regresion_Duracion_Llamadas.py", label="Abrir modelo 2")
with c3:
    st.markdown("**3 · Clasificación: llamadas no atendidas**")
    st.caption("Estima la probabilidad de que una llamada no sea atendida.")
    st.page_link("pages/3_Prediccion.py", label="Abrir modelo 3")

# ---------------------------------------------------------------- Datos agregados (sin datos personales)
vision = pd.read_csv("datos/vision_equipo.csv")
por_franja = pd.read_csv("datos/resumen_equipo_franja.csv")
perf_equipo = pd.read_csv("datos/perfiles_por_equipo.csv")
perfiles = pd.read_csv("datos/perfiles_clusters.csv")

# ---------------------------------------------------------------- Contexto por equipo
st.subheader("Contexto por equipo de trabajo")
equipo = st.selectbox("Equipo de trabajo", vision["equipo"])
f = vision[vision["equipo"] == equipo].iloc[0]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Trabajadores", f"{int(f['trabajadores'])}")
m2.metric("Llamadas registradas", f"{int(f['llamadas']):,}".replace(",", "."))
m3.metric("Llamadas no atendidas", f"{f['pct_no_atendidas']:.1f} %")
m4.metric("Duración mediana", f"{f['duracion_mediana']:.0f} s")
m5.metric("Llamadas por trabajador", f"{int(f['llamadas_por_trabajador']):,}".replace(",", "."))

g1, g2 = st.columns(2)
with g1:
    st.markdown("**Duración promedio por franja horaria** (llamadas atendidas, en segundos)")
    franja = (por_franja[por_franja["equipo"] == equipo]
              .set_index("franja")["duracion_media"].reindex(["Mañana", "Tarde", "Noche"]))
    st.bar_chart(franja)
with g2:
    st.markdown("**Perfiles de trabajadores del equipo** (clusters del modelo 1)")
    pe = perf_equipo[perf_equipo["equipo"] == equipo].set_index("perfil")["trabajadores"]
    st.bar_chart(pe, horizontal=True)

# ---------------------------------------------------------------- Comparación entre equipos
st.markdown("**Comparación entre equipos**")
tabla = vision.rename(columns={
    "equipo": "Equipo", "trabajadores": "Trabajadores", "llamadas": "Llamadas",
    "pct_no_atendidas": "% no atendidas", "duracion_media": "Duración media (s)",
    "duracion_mediana": "Duración mediana (s)", "llamadas_por_trabajador": "Llamadas por trabajador"})
st.dataframe(tabla, hide_index=True, width="stretch")

# ---------------------------------------------------------------- Perfiles
with st.expander("Descripción de los cinco perfiles de trabajadores (clusters)"):
    resumen_perfiles = perfiles[["Perfil", "Número de trabajadores", "Edad promedio", "Hijos promedio",
                                 "Antigüedad promedio", "Cargo predominante",
                                 "Homologacion Equipo de trabajo predominante"]].rename(
        columns={"Homologacion Equipo de trabajo predominante": "Equipo predominante"})
    st.dataframe(resumen_perfiles, hide_index=True, width="stretch")
    st.caption("Los perfiles son tendencias de la población caracterizada, no grupos completamente separados.")

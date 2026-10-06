import json
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Modelo 3 · Llamadas no atendidas", layout="wide")


@st.cache_resource
def cargar_modelo():
    return pickle.load(open("modelos/modelo_prediccion.pkl", "rb"))


modelo, encoders, variables = cargar_modelo()
metricas = json.load(open("modelos/metricas_prediccion.json", encoding="utf-8"))
historico = pd.read_csv("datos/no_atendidas_equipo_hora.csv")
umbral = metricas["umbral"]


def franja_de(hora):
    """La franja se deduce de la hora (misma regla que tienen los datos)."""
    if hora < 6 or hora >= 18:
        return "Noche"
    return "Mañana" if hora < 12 else "Tarde"


def preparar(df):
    """Misma preparación del notebook: LabelEncoder en las categóricas y hora numérica."""
    d = df.copy()
    for col, le in encoders.items():
        d[col] = le.transform(d[col].astype(str))
    return d[variables]


st.title("Modelo 3 · Probabilidad de que una llamada no sea atendida")
st.write(
    "Random Forest de clasificación que estima la probabilidad de que una llamada quede **abandonada o perdida**, "
    "según el equipo, el día, la hora y la dirección de la llamada. Sirve para **anticipar franjas y equipos de riesgo**."
)

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("Datos de la llamada")
    equipo = st.selectbox("Equipo de trabajo", list(encoders["Homologación Equipo de trabajo"].classes_))
    dia = st.selectbox("Día de la semana", ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"])
    hora = st.slider("Hora de inicio", min_value=0, max_value=23, value=10, step=1)
    direccion = st.radio(
        "Dirección de la llamada", ["Incoming", "Outgoing"], horizontal=True,
        format_func=lambda x: "Entrante (Incoming)" if x == "Incoming" else "Saliente (Outgoing)",
    )
    franja = franja_de(hora)
    st.caption(f"Franja horaria (según la hora): **{franja}**")


def fila_llamada(h):
    return {"Dirección de la llamada": direccion, "Homologación Equipo de trabajo": equipo,
            "Dia Semana": dia, "Hora del dia": h, "Franja horaria": franja_de(h)}


prob = float(modelo.predict_proba(preparar(pd.DataFrame([fila_llamada(hora)])))[0, 1])

with col_out:
    st.subheader("Resultado")
    r1, r2 = st.columns(2)
    r1.metric("Probabilidad de no atención", f"{prob:.0%}")
    r2.metric("Tasa global de no atención", f"{metricas['tasa_no_atendida']:.0f} %")
    st.progress(min(max(prob, 0.0), 1.0))
    if prob >= umbral:
        st.error(f"Riesgo alto: el modelo marca esta llamada como **probablemente no atendida** (umbral {umbral:.2f}).")
    else:
        st.success(f"Riesgo bajo: el modelo la marca como **probablemente atendida** (umbral {umbral:.2f}).")

    h = historico[(historico["equipo"] == equipo) & (historico["hora"] == hora)]
    if len(h) and int(h["llamadas"].iloc[0]) > 0:
        tasa = h["no_atendidas"].iloc[0] / h["llamadas"].iloc[0]
        st.metric(f"Histórico de {equipo} a las {hora}:00", f"{tasa:.0%} no atendidas",
                  f"{int(h['llamadas'].iloc[0]):,} llamadas".replace(",", "."), delta_color="off")

    st.markdown("**Probabilidad de no atención según la hora del día** (con el equipo, día y dirección elegidos)")
    horas = pd.DataFrame([fila_llamada(x) for x in range(24)])
    curva = pd.DataFrame({"hora": range(24),
                          "Probabilidad de no atención": modelo.predict_proba(preparar(horas))[:, 1]})
    st.line_chart(curva.set_index("hora"))

with st.expander("Calidad del modelo y limitaciones"):
    a, b, c, d = st.columns(4)
    a.metric("AUC", f"{metricas['auc']:.2f}")
    b.metric("Exactitud", f"{metricas['exactitud']:.2f}")
    c.metric("Precisión (no atendida)", f"{metricas['precision']:.2f}")
    d.metric("Recall (no atendida)", f"{metricas['recall']:.2f}")
    st.write(
        f"Con el umbral de {umbral:.2f}, de cada 100 alertas unas {metricas['precision'] * 100:.0f} son llamadas realmente "
        f"no atendidas, y el modelo detecta unas {metricas['recall'] * 100:.0f} de cada 100 llamadas no atendidas. "
        "Es una herramienta de apoyo para identificar franjas de riesgo, no un detector exacto de llamadas perdidas."
    )

import json
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Modelo 2 · Duración de llamadas", layout="wide")

# ---------------------------------------------------------------- Carga del modelo
@st.cache_resource
def cargar_modelo():
    return pickle.load(open("modelos/modelo_regresion_llamadas.pkl", "rb"))

modelo, min_max_scaler, variables = cargar_modelo()
metricas = json.load(open("modelos/metricas_regresion_llamadas.json", encoding="utf-8"))
resumen = pd.read_csv("datos/resumen_equipo.csv").set_index("equipo")


def franja_de(hora):
    """La franja se deduce de la hora (misma regla que tienen los datos)."""
    if hora < 6 or hora >= 18:
        return "Noche"
    return "Mañana" if hora < 12 else "Tarde"


def preparar(df):
    """Misma preparación del notebook. En despliegue: drop_first=False, reindex y scaler sin fit."""
    prep = pd.get_dummies(df, columns=["equipo", "dia", "franja", "direccion"], drop_first=False, dtype=int)
    prep = prep.reindex(columns=variables, fill_value=0)
    prep[["hora"]] = min_max_scaler.transform(prep[["hora"]])
    return prep


# ---------------------------------------------------------------- Interfaz
st.title("Modelo 2 · Duración esperada de una llamada")
st.write(
    "Regresión (Random Forest) que estima cuántos segundos dura, en promedio, una llamada contestada "
    "según su contexto. Sirve para **planear turnos**, no para predecir una llamada individual."
)

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("Datos de la llamada")
    equipo = st.selectbox("Equipo de trabajo", ["Gestión Daños", "Operación Distribución", "Operación Transmisión"])
    dia = st.selectbox("Día de la semana", ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"])
    hora = st.slider("Hora de inicio", min_value=0, max_value=23, value=10, step=1)
    direccion = st.radio(
        "Dirección de la llamada", ["Outgoing", "Incoming"], horizontal=True,
        format_func=lambda x: "Saliente (Outgoing)" if x == "Outgoing" else "Entrante (Incoming)",
    )
    franja = franja_de(hora)
    st.caption(f"Franja horaria (según la hora): **{franja}**")

# Dataframe con los mismos nombres de variables del notebook
datos = pd.DataFrame([[equipo, hora, dia, franja, direccion]],
                     columns=["equipo", "hora", "dia", "franja", "direccion"])
prediccion = float(modelo.predict(preparar(datos))[0])

with col_out:
    st.subheader("Resultado")
    r1, r2 = st.columns(2)
    r1.metric("Duración esperada", f"{prediccion:.0f} s")
    r2.metric("En minutos", f"{prediccion / 60:.1f} min")
    st.metric(f"Promedio histórico de {equipo}", f"{resumen.loc[equipo, 'duracion_media']:.0f} s")

    st.markdown("**Duración esperada según la hora del día** (con el equipo, día y dirección elegidos)")
    horas = pd.DataFrame({"hora": range(24)})
    horas["equipo"], horas["dia"], horas["direccion"] = equipo, dia, direccion
    horas["franja"] = horas["hora"].map(franja_de)
    horas["Duración esperada (s)"] = modelo.predict(preparar(horas[["equipo", "hora", "dia", "franja", "direccion"]]))
    st.line_chart(horas.set_index("hora")["Duración esperada (s)"])

with st.expander("Calidad del modelo (validación cruzada) y limitaciones"):
    a, b, c = st.columns(3)
    a.metric("MAE", f"{metricas['MAE']:.0f} s", f"base: {metricas['MAE_base']:.0f} s", delta_color="off")
    b.metric("RMSE", f"{metricas['RMSE']:.0f} s")
    c.metric("R²", f"{metricas['R2']:.2f}")
    st.write(
        "El modelo mejora muy poco a la línea base (predecir siempre el promedio) y explica cerca del 3 % "
        "de la variabilidad: el dataset no trae el motivo de la llamada. Úsalo como estimación de la "
        "duración promedio por segmento y reentrénalo periódicamente."
    )

# App Streamlit · Proyecto de minería de datos (Centro de control de energía)

App del grupo con **tres modelos**, una página por modelo, más una página de inicio informativa.

| Página | Archivo | Responsable | Modelo |
|---|---|---|---|
| Inicio (contexto por equipo de trabajo) | `app.py` | Grupo | — |
| 1 · Clustering de trabajadores | `pages/1_Clustering.py` | Daniela Alejandra Vergara Pereira | K-Means (5 perfiles) |
| 2 · Regresión: duración de llamadas | `pages/2_Regresion_Duracion_Llamadas.py` | Sebastián Ramírez Maldonado | Random Forest |
| 3 · Clasificación: llamadas no atendidas | `pages/3_Prediccion.py` | Wilmer Andrés Ortiz Bautista | Random Forest |

## Estructura del repositorio

```
app.py                        # página de inicio (informativa)
pages/                        # una página por modelo
modelos/                      # [modelo, preprocesamiento, variables] + medidas en .json
datos/                        # tablas agregadas (sin nombres ni IDs)
requirements.txt
README.md
```

`app.py` debe quedar en la **raíz** del repositorio.

## Ejecutar en el computador

```
pip install -r requirements.txt
streamlit run app.py
```

## Publicar (GitHub + Streamlit Community Cloud)

1. Subir el contenido de esta carpeta a un repositorio de GitHub, todo en la rama `main`.
2. En share.streamlit.io: *Create app* → elegir el repositorio, la rama `main` y el archivo `app.py`.
3. En *Advanced settings* elegir Python 3.12 o 3.13.
4. Tomar el pantallazo de la app publicada para la rúbrica.

## Regenerar los modelos

Ejecutar el notebook `Proyecto_mineria_de_datos.ipynb` completo: cada modelo termina con una celda que guarda su `.pkl`
y sus medidas en `app_streamlit/modelos/`, y las tablas agregadas en `app_streamlit/datos/`.

## Importante: versiones

Un `.pkl` solo se carga bien con la **misma versión de scikit-learn** con la que se creó (aquí, la 1.8.0).
Si ejecutan el notebook en otro entorno (por ejemplo Colab), actualicen la línea `scikit-learn==...` de
`requirements.txt` con la versión que imprime `import sklearn; sklearn.__version__` y suban los `.pkl` nuevos.

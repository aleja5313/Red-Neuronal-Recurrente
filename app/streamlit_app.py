"""
App de Streamlit: Clasificador de sentimiento de reseñas (LSTM)
------------------------------------------------------------------
Carga un modelo LSTM previamente entrenado (modelo_sentimiento_imdb.keras) y permite
al usuario escribir una reseña de película para predecir si es positiva o negativa,
junto con el nivel de confianza de la predicción.

Para correrla localmente:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf

# ----------------------------------------------------------------------
# Rutas al modelo y a la lista de clases (deben existir en la carpeta models/)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_sentimiento_imdb.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

NOMBRES_BONITOS = {
    "pos": "Positiva 😊",
    "neg": "Negativa 😞",
}

EJEMPLOS = [
    "This movie was absolutely wonderful, the acting and story were fantastic!",
    "What a waste of time, the plot made no sense and the acting was terrible.",
]
@tf.keras.utils.register_keras_serializable()
def limpiar_texto(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, "<br />", " ")

@st.cache_resource
def cargar_modelo_y_clases():
    """Carga el modelo entrenado (incluye la vectorización de texto) una sola vez."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    modelo = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"limpiar_texto": limpiar_texto}
    )
    with open(CLASS_NAMES_PATH, "r") as f:
        clases = json.load(f)
    return modelo, clases

def main():
    st.set_page_config(page_title="Clasificador de Sentimiento (LSTM)", page_icon="🎬", layout="centered")

    st.title("🎬 Clasificador de Sentimiento de Reseñas")
    st.write(
        "Escribe (en inglés) una reseña de película y el modelo te dirá si cree que es "
        "**positiva o negativa**, junto con su nivel de confianza."
    )

    modelo, clases = cargar_modelo_y_clases()

    if modelo is None:
        st.error(
            "No se encontró el modelo entrenado en `models/modelo_sentimiento_imdb.keras`.\n\n"
            "Entrena el modelo con el notebook `notebooks/RNN_IMDB_Sentimiento.ipynb` en Google Colab, "
            "descarga `modelo_sentimiento_imdb.keras` y `class_names.json`, y colócalos dentro de la carpeta "
            "`models/` de este repositorio antes de ejecutar la app."
        )
        st.stop()

    with st.expander("Ver ejemplos de reseñas que puedes probar"):
        for ejemplo in EJEMPLOS:
            st.write(f"- {ejemplo}")

    texto = st.text_area(
        "Escribe tu reseña aquí (en inglés, el modelo se entrenó con reseñas en inglés):",
        height=150,
        placeholder="Type a movie review here...",
    )

    if st.button("Analizar sentimiento", type="primary"):
        if not texto.strip():
            st.warning("Escribe una reseña antes de analizar.")
        else:
            with st.spinner("Analizando el texto..."):
                entrada = tf.constant([texto])
                probabilidad = float(modelo.predict(entrada)[0][0])
                clase_predicha = "pos" if probabilidad > 0.5 else "neg"
                confianza = (probabilidad if probabilidad > 0.5 else 1 - probabilidad) * 100

            nombre_mostrado = NOMBRES_BONITOS.get(clase_predicha, clase_predicha)
            st.success(f"**Sentimiento predicho: {nombre_mostrado}**")
            st.metric(label="Nivel de confianza", value=f"{confianza:.1f}%")
            st.progress(confianza / 100)
    else:
        st.info("Escribe una reseña y presiona el botón para analizarla.")


if __name__ == "__main__":
    main()

# Clasificador de Sentimiento con RNN/LSTM (Deep Learning)

Proyecto de clasificación de texto usando una red neuronal recurrente (LSTM bidireccional)
entrenada desde cero con el dataset **IMDB Reviews** (reseñas de películas en inglés,
positiva/negativa). Incluye el notebook de experimentación completo y una aplicación en
Streamlit para probar el modelo con texto nuevo.

## Estructura del repositorio

```
proyecto-rnn-imdb-sentimiento/
├── notebooks/
│   └── RNN_IMDB_Sentimiento.ipynb   # Notebook completo: EDA, preprocesamiento, entrenamiento, evaluación
├── app/
│   └── streamlit_app.py             # Aplicación Streamlit de predicción
├── models/
│   ├── modelo_sentimiento_imdb.keras  # Modelo entrenado (se genera al correr el notebook)
│   └── class_names.json               # Nombres de las clases, en el orden que usa el modelo
├── requirements.txt
└── README.md
```

## Dataset

**IMDB Reviews** (Stanford AI Lab, descarga pública sin necesidad de cuenta de Kaggle):
50.000 reseñas de películas en inglés, balanceadas 50/50 entre positivas y negativas
(25.000 para entrenamiento, 25.000 para prueba).

---

## 1. Cómo reproducir el entrenamiento (Google Colab)

1. Abre `notebooks/RNN_IMDB_Sentimiento.ipynb` en Google Colab.
2. Activa GPU: `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` → `GPU (T4)`.
3. Ejecuta las celdas en orden (el dataset se descarga solo, no necesita credenciales).
4. Al final del notebook, el modelo se guarda y se descarga automáticamente como
   `modelo_sentimiento_imdb.keras` junto con `class_names.json`.
5. Coloca esos dos archivos descargados dentro de la carpeta `models/` de este repositorio
   (reemplazando los que haya, si existen).

## 2. Cómo correr la aplicación Streamlit en tu computador

Requiere Python 3.10+ y que ya existan `models/modelo_sentimiento_imdb.keras` y
`models/class_names.json` (ver paso anterior).

1. **Descomprime el proyecto** en tu computador, si aún no lo has hecho. Debe quedar la carpeta
   `proyecto-rnn-imdb-sentimiento` con las subcarpetas `notebooks`, `app` y `models`.

2. **Verifica que el modelo esté en su lugar**: dentro de `models/` deben estar
   `modelo_sentimiento_imdb.keras` y `class_names.json`. Sin ellos, la app no puede cargar el modelo.

3. **Abre una terminal** en la carpeta del proyecto:
   ```bash
   cd ruta/a/proyecto-rnn-imdb-sentimiento
   ```

4. **(Recomendado) Crea un entorno virtual**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

5. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Ejecuta la app**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Se abrirá automáticamente una pestaña del navegador, normalmente en `http://localhost:8501`.

7. **Prueba la app**: escribe una reseña de película en inglés (hay ejemplos dentro de la app)
   y verás el sentimiento predicho junto con el porcentaje de confianza.

## 3. Cómo subir este proyecto a GitHub

Desde la carpeta `proyecto-rnn-imdb-sentimiento` (donde ya está inicializado el repositorio
Git local):

```bash
git remote add origin <URL-DE-TU-REPOSITORIO>.git
git branch -M main
git push -u origin main
```

**Si el push es rechazado** (mensaje de "rejected" / "fetch first"), normalmente es porque el
repositorio en GitHub ya tiene algo (por ejemplo, lo creaste marcando "Add a README"):

```bash
git pull origin main --allow-unrelated-histories
# Si aparece un conflicto en README.md, ábrelo, deja la versión que prefieras, guarda, y luego:
git add README.md
git commit -m "Resolver conflicto de README"
git push -u origin main
```

Para subir cambios futuros:

```bash
git add -A
git commit -m "Describe aquí qué cambiaste"
git push
```

## Resumen técnico del modelo

- Arquitectura: `Embedding` (64 dimensiones) + dos capas `Bidirectional(LSTM)` + capas densas
  con `Dropout`, construida desde cero.
- Preprocesamiento: limpieza de HTML y minúsculas, luego `TextVectorization` (vocabulario de
  10.000 palabras, secuencias de 250 tokens). El vectorizador queda **incluido dentro del
  modelo guardado**, así que la app puede enviarle texto crudo directamente.
- Salida: una neurona con activación `sigmoid` (clasificación binaria positiva/negativa).
- Optimizador: Adam. Función de pérdida: binary crossentropy.
- Callbacks: EarlyStopping y ModelCheckpoint sobre la pérdida/exactitud de validación.
- Evaluación: accuracy, precision/recall/F1, matriz de confusión, y mínimo 10 predicciones
  sobre reseñas reales del conjunto de prueba, con el texto y el nivel de confianza.

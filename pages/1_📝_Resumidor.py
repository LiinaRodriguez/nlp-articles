import streamlit as st
import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import re
import time
import os
import unicodedata

# Configuración de la página
st.set_page_config(page_title="Generador de Resúmenes", page_icon="🧠", layout="wide")

# Definir el ID del modelo en Hugging Face Hub
MODEL_ID = "liinarodriguez/summarization"  # Modelo público en Hugging Face Hub


# ======== Cargar modelo y tokenizer desde Hugging Face ========
@st.cache_resource
def load_model():
    try:
        tokenizer = BartTokenizer.from_pretrained(MODEL_ID)
        model = BartForConditionalGeneration.from_pretrained(MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error al cargar el modelo desde Hugging Face: {str(e)}")
        st.stop()


# ======== Función para limpiar texto ========
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Normaliza unicode (acentos, etc)
    text = unicodedata.normalize("NFKC", text)

    # Reemplaza saltos de línea, tabs por espacio
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Elimina caracteres no imprimibles (control)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    # Elimina caracteres raros excepto letras, números, signos básicos y espacios
    text = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜñÑ.,;:()\-\'\" ]+", " ", text)

    # Normaliza múltiples espacios a uno solo
    text = re.sub(r"\s+", " ", text)

    # Recorta espacios al inicio y final
    text = text.strip()

    return text


# ======== Función para resumir un texto ========
def summarize(text):
    try:
        text = clean_text(text)
        if not text:
            return "Error: Texto vacío"

        # Tokenización y limpieza del texto
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            padding="max_length",
            truncation=True,
        ).to(device)

        # Generación del resumen
        with torch.no_grad():
            summary_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=350,  # Como en el entrenamiento
                num_beams=10,  # Como en el notebook
                length_penalty=1.2,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        st.error(f"Error al generar el resumen: {str(e)}")
        return f"Error al generar el resumen: {str(e)}"


# ======== Cargar el modelo ==========
with st.spinner("Cargando el modelo..."):
    model, tokenizer, device = load_model()
    st.success("✅ Modelo cargado correctamente")

# ======== Interfaz Streamlit ========
st.title(" Generador de Resúmenes con Modelo Fine-tuneado")
st.markdown(
    """
Este modelo ha sido entrenado específicamente para generar resúmenes de textos científicos.
Puedes ingresar texto directamente, subir un archivo .txt o un archivo .csv con múltiples textos.
"""
)


# Selector de modo de entrada
input_mode = st.radio(
    "Selecciona el modo de entrada:",
    ["✏️ Texto directo", "📂 Archivo .txt", "📊 Archivo .csv"],
)

if input_mode == "✏️ Texto directo":
    text_input = st.text_area("Ingresa el texto a resumir:", height=200)
    col1, col2 = st.columns(2)
    if col1.button("✍️ Generar resumen"):
        if text_input:
            with st.spinner("Generando resumen..."):
                resumen = summarize(text_input)
                st.success("✅ Resumen generado:")
                st.write(resumen)

                # Mostrar métricas
                col1, col2 = st.columns(2)
                col1.metric("Longitud del texto original", len(text_input.split()))
                col2.metric("Longitud del resumen", len(resumen.split()))
        else:
            st.warning("⚠️ Por favor, ingresa un texto para resumir")

elif input_mode == "📂 Archivo .txt":
    uploaded_file = st.file_uploader("Sube un archivo .txt", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.text_area("📜 Texto cargado:", text, height=200)

        if st.button("✍️ Generar resumen"):
            with st.spinner("Resumiendo..."):
                resumen = summarize(text)
                st.success("✅ Resumen generado:")
                st.write(resumen)

                # Mostrar métricas
                col1, col2 = st.columns(2)
                col1.metric("Longitud del texto original", len(text.split()))
                col2.metric("Longitud del resumen", len(resumen.split()))

elif input_mode == "📊 Archivo .csv":
    uploaded_file = st.file_uploader("Sube un archivo .csv", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📑 Vista previa del archivo:")
            st.dataframe(df.head())

            col_name = st.selectbox(
                "Selecciona la columna con los textos a resumir:", df.columns
            )

            if st.button("✍️ Generar resúmenes"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                total_rows = len(df)
                resumes = []

                for i, text in enumerate(df[col_name]):
                    resumen = summarize(text)
                    resumes.append(resumen)
                    progress = (i + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"Procesando {i+1} de {total_rows} textos...")
                    time.sleep(0.1)  # Para evitar sobrecarga

                df["Resumen"] = resumes
                st.success("✅ Resúmenes generados")

                # Mostrar los resultados en una tabla con scroll
                st.markdown("### Resultados:")
                st.dataframe(
                    df[[col_name, "Resumen"]], height=400, use_container_width=True
                )

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Descargar CSV con resúmenes",
                    data=csv,
                    file_name="resumenes.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error al procesar el archivo CSV: {str(e)}")

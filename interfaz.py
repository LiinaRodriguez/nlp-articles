import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import os 

# Configuración de página
st.set_page_config(
    page_title="PLN para Investigación Veterinaria",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cargar el modelo
@st.cache_resource
def load_model():
    MODEL_ID = "liinarodriguez/summarization"
    try:
        tokenizer = BartTokenizer.from_pretrained(MODEL_ID)
        model = BartForConditionalGeneration.from_pretrained(MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()


# Función para resumir
def generate_summary(text, model, tokenizer, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        padding="max_length",
        truncation=True,
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=350,
            num_beams=10,
            length_penalty=1.2,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Cargar el modelo al inicio
model, tokenizer, device = load_model()

# Estilos CSS personalizados
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
    
    :root {
        --primary: #4e89ae;
        --secondary: #ed6663;
        --accent: #ffa372;
        --light: #f8f9fa;
        --dark: #43658b;
    }
    
    * {
        font-family: 'Montserrat', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    }
    
    .title-box {
        background: linear-gradient(90deg, var(--primary) 0%, var(--dark) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--secondary);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .highlight {
        background: linear-gradient(120deg, rgba(237,102,99,0.2), rgba(237,102,99,0));
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    
    .progress-bar {
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent), var(--secondary));
        transition: width 0.5s ease;
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: var(--dark);
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    .floating {
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }
</style>
""",
    unsafe_allow_html=True,
)

# Título con diseño mejorado
st.markdown(
    '<div class="title-box"><h1 style="color:white; margin:0;">Paula y su Investigación </h1></div>',
    unsafe_allow_html=True,
)

# Sección 1: Introducción con efecto de máquina de escribir
st.header("El reto de Paula", anchor="paula-challenge")

with st.expander("Ver la historia completa", expanded=True):
    challenge_text = """
    Paula es una estudiante de Medicina Veterinaria que se enfrenta al gran desafío de su trabajo de grado: 
    investigar el cáncer de mama en perras. Tiene una carpeta con más de 100 artículos científicos, 
    pero poco tiempo para leerlos todos. Necesita encontrar rápidamente las tendencias actuales y extraer 
    información clave para su investigación.
    """

    # Simulación de máquina de escribir
    placeholder = st.empty()
    full_text = challenge_text
    current_text = ""

    for char in full_text:
        current_text += char
        placeholder.markdown(
            f'<div class="card"><p style="font-size:1.1rem;">{current_text}</p></div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.03)

    # Imagen ilustrativa con efecto flotante
    st.markdown(
        '<div class="floating" style="text-align:center; margin:2rem 0;">',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# Sección 2: Descubre la app con simulación interactiva
st.header("Descubre la herramienta de PLN", anchor="discover-tool")

st.markdown(
    """
<div class="card">
    <p style="font-size:1.1rem;">
    Un día, Paula encuentra una aplicación que promete algo increíble: 
    <span class="highlight">resumir artículos científicos automáticamente</span> y mostrar los 
    <span class="highlight">temas más comunes</span> con solo arrastrar sus archivos.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Simulación de carga de archivos
st.subheader("Simula la experiencia de Paula")
st.markdown(
    """
<div class="card">
    <p>Selecciona el modo de entrada que quieres probar:</p>
</div>
""",
    unsafe_allow_html=True,
)

input_type = st.radio(
    "",
    [
        "📚 Usar artículos de ejemplo",
        "📂 Subir mis propios archivos",
    ],
)

if input_type == "📚 Usar artículos de ejemplo":
    st.info("Estos son algunos artículos de muestra sobre oncología veterinaria")

    # Leer los archivos de ejemplo
    sample_files = {}
    sample_path = os.path.join(os.path.dirname(__file__), "samples")

    for filename in os.listdir(sample_path):
        if filename.endswith(".txt"):
            with open(os.path.join(sample_path, filename), "r", encoding="utf-8") as f:
                sample_files[filename] = f.read()

    # Selector de archivo
    selected_file = st.selectbox(
        "Selecciona un artículo para analizar:", list(sample_files.keys())
    )

    if selected_file:
        text = sample_files[selected_file]
        st.markdown(
            """
        <div class="card">
            <h4>Contenido del artículo</h4>
            <div style="max-height: 300px; overflow-y: auto; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;">
            {}</div>
        </div>
        """.format(
                text.replace("\n", "<br>")
            ),
            unsafe_allow_html=True,
        )

        if st.button("✨ Generar resumen"):
            with st.spinner("Analizando el artículo..."):
                resumen = generate_summary(text, model, tokenizer, device)

                st.markdown(
                    """
                <div class="card">
                    <h4>Resumen generado 📝</h4>
                    <div style="background-color:#e8f4f8; padding:1rem; border-radius:8px; border-left:4px solid var(--primary);">
                    {}</div>
                </div>
                """.format(
                        resumen
                    ),
                    unsafe_allow_html=True,
                )

                # Métricas
                col1, col2, col3 = st.columns(3)
                words_original = len(text.split())
                words_summary = len(resumen.split())
                reduction = ((words_original - words_summary) / words_original) * 100

                col1.metric("Palabras en original", words_original)
                col2.metric("Palabras en resumen", words_summary)
                col3.metric("Reducción", f"{reduction:.1f}%")

elif input_type == "📂 Subir mis propios archivos":
    uploaded_files = st.file_uploader(
        "Arrastra tus archivos aquí",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} archivos cargados exitosamente!")

        # Barra de progreso simulada
        st.markdown(
            '<div class="card"><p>Analizando documentos...</p></div>',
            unsafe_allow_html=True,
        )
        progress_bar = st.empty()
        progress_text = st.empty()

        for percent in range(0, 101, 5):
            progress_bar.markdown(
                f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width:{percent}%"></div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            progress_text.text(f"Progreso: {percent}%")
            time.sleep(0.1)

        st.balloons()
        st.success("Análisis completado! Ver resultados abajo")

# Sección 3: Resultados visuales interactivos
st.header("¿Qué encuentra Paula?", anchor="paula-findings")

# Gráfico de temas con selección interactiva
st.markdown(
    """
<div class="card">
    <p>Después de cargar sus artículos, la herramienta le presenta los siguientes tópicos más frecuentes:</p>
</div>
""",
    unsafe_allow_html=True,
)

topics = {
    "Topic 1: Biología molecular y procesos celulares": 22,
    "Topic 2 Biotecnología aplicada": 11,
    "Topic 3 Inteligencia artificial en medicina": 24,
    "Topic 4 Patología clínica e histológica": 28,
    "Topic 5 Análisis de imágenes médicas": 25,
}

# Convertir a DataFrame para mejor visualización
df_topics = pd.DataFrame(
    {"Tema": list(topics.keys()), "Artículos": list(topics.values())}
)

# Gráfico interactivo
st.subheader("Distribución de temas en la literatura")
tab1, tab2= st.tabs(["Gráfico de Barras", ""])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.PuBu(np.linspace(0.4, 1, len(topics)))
    df_topics.sort_values("Artículos", ascending=True).plot.barh(
        x="Tema", y="Artículos", ax=ax, color=colors
    )
    ax.set_title("Tópicos más frecuentes en oncología veterinaria", fontsize=14)
    ax.set_xlabel("Número de artículos", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    st.pyplot(fig)


# Sección 4: Impacto con visualización de métricas
st.header("Impacto para Paula", anchor="impact")

cols = st.columns(2)
cols[0].metric("Artículos procesados", "110", "100% de su colección")
cols[1].metric("Temas identificados", "5", "Tendencias clave")

st.markdown(
    """
<div class="card">
    <h4>Gracias a la herramienta, Paula:</h4>
    <ul>
        <li>Redujo el tiempo de revisión de literatura en más de un 60%</li>
        <li>Identificó tendencias clave para enfocar su investigación</li>
        <li>Sintetizó información técnica sin perder calidad</li>
        <li>Pudo dedicar más tiempo al análisis y redacción</li>
        <li>Descubrió conexiones entre estudios que había pasado por alto</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

# Sección final con conclusión interactiva
st.header("Conclusión", anchor="conclusion")

st.markdown(
    """
<div class="card">
    <div style="text-align:center; padding:2rem;">
        <h3 style="color:var(--dark);">Transformando datos en conocimiento</h3>
        <p style="font-size:1.2rem;">
        Esta es la historia de cómo el <span class="highlight">Procesamiento de Lenguaje Natural</span> 
        y <span class="highlight">Streamlit</span> ayudaron a Paula a transformar montañas de datos 
        en conocimiento accionable para su investigación.
        </p>
        <div style="margin:2rem 0;">
            <img src="https://images.unsplash.com/photo-1558494940-1b7b0b7a5d9a" 
                 width="300" style="border-radius:50%; border:4px solid var(--accent);">
        </div>
        <button style="background:linear-gradient(90deg, var(--accent), var(--secondary)); 
                      color:white; border:none; padding:0.8rem 2rem; border-radius:30px; 
                      font-size:1.1rem; cursor:pointer; margin-top:1rem;">
            Comienza tu propia investigación!
        </button>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Notas finales
st.markdown("---")
st.caption(
    "Herramienta educativa desarrollada para demostrar el poder del PLN en investigación científica. "
   
)

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import plotly.graph_objects as go

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import DepthwiseConv2D

# Custom object pour DepthwiseConv2D (ignore l'argument 'groups' si nécessaire)
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)  # supprime l'argument non reconnu
        super().__init__(*args, **kwargs)

# Configuration de la page
st.set_page_config(
    page_title="Classification",
    page_icon="📷",
    layout="wide"
)

# Titre
st.markdown("""
<div style="text-align: center;">
    <h1 style="color: #4CAF50;">🔍 Classification des Tumeurs Cérébrales</h1>
    <p style="color: #64748b;">Chargez une image d'IRM pour identifier le type de tumeur</p>
</div>
""", unsafe_allow_html=True)

# Labels des classes
LABELS = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
LABELS_FR = {
    'glioma_tumor': 'Gliome',
    'no_tumor': 'Aucune tumeur',
    'meningioma_tumor': 'Méningiome',
    'pituitary_tumor': 'Tumeur pituitaire'
}

# Fonction de prétraitement
def preprocess_image(image, target_size=(150, 150)):
    img = np.array(image)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Upload
st.markdown("### 📤 Téléchargez une image d'IRM")
uploaded_file = st.file_uploader(
    "Sélectionnez une image (PNG, JPG, JPEG)",
    type=['png', 'jpg', 'jpeg']
)

# Chargement du modèle (une seule fois)
@st.cache_resource
def load_effnet_model():
    try:
        model = load_model("models/effnet.h5", custom_objects={"DepthwiseConv2D": FixedDepthwiseConv2D})
        return model
    except Exception as e:
        st.error(f"Impossible de charger le modèle: {e}")
        return None

model = load_effnet_model()

# Si image uploadée
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Image originale**")
        st.image(image, width=350)
    
    if st.button("🔬 Analyser l'image", use_container_width=True, type="secondary"):
        with st.spinner("Analyse en cours..."):
            try:
                img_array = preprocess_image(image)
                predictions = model.predict(img_array, verbose=0)[0]
                pred_class_index = np.argmax(predictions)
                pred_class = LABELS[pred_class_index]
                pred_percent = predictions * 100
                
                st.session_state.predictions = pred_percent
                st.session_state.pred_class = pred_class
                st.session_state.image_array = np.array(image)
                
                st.success("✅ Analyse terminée!")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
                st.session_state.predictions = None

# Affichage des résultats
if hasattr(st.session_state, 'predictions') and st.session_state.predictions is not None:
    st.markdown("### 📊 Résultats de la classification")
    
    pred_class = st.session_state.pred_class
    pred_percent = st.session_state.predictions
    confidence = pred_percent[np.argmax(pred_percent)]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if pred_class == 'no_tumor':
            color = "green"
            emoji = "✅"
            message = "Aucune tumeur détectée"
        else:
            color = "orange" if confidence < 80 else "red"
            emoji = "⚠️" if confidence < 80 else "🚨"
            message = f"{LABELS_FR[pred_class]} détecté"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; border-radius: 15px; background-color: {color}20; border: 2px solid {color}80;">
            <h2 style="color: {color}; margin-bottom: 1rem;">{emoji} {message}</h2>
            <h1 style="color: {color}; font-size: 3rem;">{confidence:.1f}%</h1>
            <p style="color: #64748b;">Confiance de la prédiction</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique probabilités
    st.markdown("#### 📈 Probabilités par classe")
    fig = go.Figure(data=[go.Bar(
        x=[LABELS_FR[label] for label in LABELS],
        y=pred_percent,
        marker_color=['#FF6B6B' if i == np.argmax(pred_percent) else '#4ECDC4' for i in range(len(LABELS))],
        text=[f"{p:.1f}%" for p in pred_percent],
        textposition='outside'
    )])
    fig.update_layout(title="Distribution des probabilités", yaxis_title="Probabilité (%)", xaxis_title="Type de tumeur", height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Rapport texte
    st.markdown("### 📄 Rapport d'analyse")
    col1, col2 = st.columns(2)
    with col1:
        report = f"""
        RAPPORT D'ANALYSE - CLASSIFICATION DE TUMEUR CÉRÉBRALE
        
        Résultat: {LABELS_FR[pred_class]}
        Confiance: {confidence:.1f}%
        
        Détails des probabilités:
        - Gliome: {pred_percent[0]:.1f}%
        - Aucune tumeur: {pred_percent[1]:.1f}%
        - Méningiome: {pred_percent[2]:.1f}%
        - Tumeur pituitaire: {pred_percent[3]:.1f}%
        
        Note: Ce rapport est généré automatiquement par un système d'IA.
        Consultez toujours un professionnel de santé pour un diagnostic définitif.
        """
        st.download_button("📥 Télécharger le rapport (TXT)", data=report, file_name="rapport_classification.txt", mime="text/plain", use_container_width=True)
    
    # Image annotée
    with col2:
        if st.button("🖼️ Générer l'image annotée", use_container_width=True):
            img_annotated = Image.fromarray(st.session_state.image_array)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img_annotated)
            text = f"Diagnostic: {LABELS_FR[pred_class]} ({confidence:.1f}%)"
            draw.text((10, 10), text, fill=(255, 0, 0))
            temp_path = tempfile.mktemp(suffix='.png')
            img_annotated.save(temp_path)
            with open(temp_path, "rb") as f:
                st.download_button("📥 Télécharger l'image annotée", data=f, file_name="image_annotee.png", mime="image/png", use_container_width=True)
            os.unlink(temp_path)

# Instructions si aucune image
else:
    st.info("""
    ### 📋 Instructions pour la classification:
    
    1. Téléchargez une image d'IRM cérébrale
    2. Formats acceptés: PNG, JPG, JPEG
    3. Image axiale de préférence, bonne résolution et contraste
    4. Cliquez sur "Analyser l'image"
    """)

st.markdown("---")
if st.button("🏠 Retour à l'accueil", use_container_width=True):
    st.switch_page("app.py")

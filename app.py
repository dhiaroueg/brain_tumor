import streamlit as st
import pandas as pd
from datetime import date
import json
import os

# Configuration de la page
st.set_page_config(
    page_title="Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E3A8A;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .title-text {
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .subtitle-text {
        font-size: 1.2rem;
        color: #f8fafc;
        opacity: 0.9;
    }
    
    .card {
        padding: 2rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .feature-card {
        height: 300px;
        transition: transform 0.3s;
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        font-weight: 600;
        font-size: 1.1rem;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
    }
    
    .btn-secondary {
        background: #4CAF50;
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        font-weight: 600;
        font-size: 1.1rem;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
    }
    
    .input-field {
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s;
    }
    
    .input-field:focus {
        border-color: #667eea;
        outline: none;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown("""
<div class="main-header">
    <div class="title-text">🧠 BRAIN TUMOR ANALYSIS</div>
    <div class="subtitle-text">Système intelligent de segmentation et classification des tumeurs cérébrales</div>
</div>
""", unsafe_allow_html=True)

# Formulaire patient
st.markdown("### 📋 Informations du Patient")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        nom = st.text_input("Nom *", placeholder="Votre nom", key="nom")
        prenom = st.text_input("Prénom *", placeholder="Votre prénom", key="prenom")
        email = st.text_input("Email *", placeholder="email@example.com", key="email")
    
    with col2:
        age = st.number_input("Âge *", min_value=0, max_value=120, step=1, key="age")
        telephone = st.text_input("Téléphone *", placeholder="+212 6 XX XX XX XX", key="telephone")
        adresse = st.text_area("Adresse *", placeholder="Votre adresse complète", height=100, key="adresse")
    
    commentaires = st.text_area(
        "Informations supplémentaires",
        placeholder="Décrivez vos symptômes, antécédents médicaux ou toute autre information pertinente...",
        height=150,
        key="commentaires"
    )
    
    # Validation
    submitted = st.form_submit_button("✅ Enregistrer les informations", use_container_width=True)
    
    if submitted:
        if not all([nom, prenom, email, age, telephone, adresse]):
            st.error("⚠️ Veuillez remplir tous les champs obligatoires (*)")
        else:
            # Sauvegarde des données (vous pourriez sauvegarder dans une base de données)
            patient_data = {
                "nom": nom,
                "prenom": prenom,
                "email": email,
                "age": age,
                "telephone": telephone,
                "adresse": adresse,
                "commentaires": commentaires,
                "date": str(date.today())
            }
            
            # Sauvegarde en local (pour démonstration)
            os.makedirs("data", exist_ok=True)
            with open(f"data/patient_{nom}_{prenom}.json", "w") as f:
                json.dump(patient_data, f, indent=4)
            
            st.success(f"✅ Informations enregistrées pour {prenom} {nom}")

# Séparateur
st.markdown("---")

# Sélection des fonctionnalités
st.markdown("### 🔬 Sélectionnez le type d'analyse")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card feature-card">
        <h3 style="color: #667eea;">🧬 Segmentation 3D</h3>
        <p style="color: #64748b; font-size: 1rem;">
        <b>Analyse approfondie des volumes cérébraux</b><br><br>
        • Segmentation multi-classes (nécrose, œdème, zone active)<br>
        • Mesures physiques précises (surface, périmètre)<br>
        • Visualisation 3D interactive<br>
        • Compatible fichiers .nii / .nii.gz
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📁 Lancer la Segmentation", use_container_width=True, type="primary"):
        st.switch_page("pages/Segmentation.py")

with col2:
    st.markdown("""
    <div class="card feature-card">
        <h3 style="color: #4CAF50;">🔍 Classification</h3>
        <p style="color: #64748b; font-size: 1rem;">
        <b>Identification du type de tumeur</b><br><br>
        • Classification en 4 catégories<br>
        • Analyse par IA EfficientNet<br>
        • Prédictions avec scores de confiance<br>
        • Compatible images PNG, JPG, JPEG
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📷 Lancer la Classification", use_container_width=True, type="secondary"):
        st.switch_page("pages/Classification.py")

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0;">
    <p>🧠 <b>Brain Tumor Analysis System</b> - Version 1.0</p>
    <p style="font-size: 0.9rem;">© 2024 Système d'aide au diagnostic médical - Pour usage professionnel uniquement</p>
</div>
""", unsafe_allow_html=True)
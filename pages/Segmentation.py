import streamlit as st
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import tempfile
import os
import plotly.graph_objects as go
import pandas as pd

# =========================
# CONFIGURATION DE LA PAGE
# =========================
st.set_page_config(
    page_title="Segmentation 3D",
    page_icon="📁",
    layout="wide"
)

st.markdown("""
<div style="text-align: center;">
    <h1 style="color: #667eea;">🧬 Segmentation 3D des Tumeurs Cérébrales</h1>
    <p style="color: #64748b;">Chargez vos fichiers d'IRM pour une analyse détaillée</p>
</div>
""", unsafe_allow_html=True)

# =========================
# PARAMÈTRES
# =========================
IMG_SIZE = 128
VOLUME_START_AT = 22
VOLUME_SLICES = 100

# =========================
# SESSION STATE
# =========================
if "segmentation_done" not in st.session_state:
    st.session_state.segmentation_done = False
if "mask" not in st.session_state:
    st.session_state.mask = None
if "flair_volume" not in st.session_state:
    st.session_state.flair_volume = None
if "t1ce_volume" not in st.session_state:
    st.session_state.t1ce_volume = None
if "dx_dy" not in st.session_state:
    st.session_state.dx_dy = (1, 1)

# =========================
# FONCTIONS
# =========================
def calculer_mesures_physiques(mask_slice, dx, dy):
    binary = (mask_slice > 0).astype(np.uint8)

    surface_mm2 = np.sum(binary) * dx * dy

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimetre_pixels = sum(cv2.arcLength(cnt, True) for cnt in contours)
    perimetre_mm = perimetre_pixels * dx

    surface_totale_mm2 = binary.shape[0] * binary.shape[1] * dx * dy
    densite = surface_mm2 / surface_totale_mm2 if surface_totale_mm2 > 0 else 0

    return surface_mm2, perimetre_mm, densite


def load_and_preprocess_data(flair_path, t1ce_path):
    flair_img = nib.load(flair_path)
    t1ce_img = nib.load(t1ce_path)

    flair = flair_img.get_fdata()
    t1ce = t1ce_img.get_fdata()

    dx, dy, dz = flair_img.header.get_zooms()

    X = np.zeros((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)

    for i in range(VOLUME_SLICES):
        X[i, :, :, 0] = cv2.resize(
            flair[:, :, i + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)
        )
        X[i, :, :, 1] = cv2.resize(
            t1ce[:, :, i + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)
        )

    X = X / np.max(X)

    return X, flair, t1ce, (dx, dy)


# =========================
# UPLOAD DES FICHIERS
# =========================
st.markdown("### 📤 Téléchargement des fichiers")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**FLAIR (.nii / .nii.gz)**")
    flair_file = st.file_uploader(
        "",
        type=["nii", "nii.gz"],
        key="flair_uploader"
    )

with col2:
    st.markdown("**T1CE (.nii / .nii.gz)**")
    t1ce_file = st.file_uploader(
        "",
        type=["nii", "nii.gz"],
        key="t1ce_uploader"
    )

# =========================
# SEGMENTATION
# =========================
if flair_file and t1ce_file:
    if st.button("🚀 Lancer la segmentation", type="primary", use_container_width=True):
        with st.spinner("Traitement en cours..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as f:
                f.write(flair_file.read())
                flair_path = f.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as f:
                f.write(t1ce_file.read())
                t1ce_path = f.name

            X, flair, t1ce, voxel_size = load_and_preprocess_data(
                flair_path, t1ce_path
            )

            import tensorflow as tf
            from tensorflow import keras

            model = keras.models.load_model(
                "models/model_x81_dcs65.h5",
                compile=False
            )

            pred = model.predict(X, verbose=0)
            mask = np.argmax(pred, axis=-1)

            st.session_state.segmentation_done = True
            st.session_state.mask = mask
            st.session_state.flair_volume = flair
            st.session_state.t1ce_volume = t1ce
            st.session_state.dx_dy = voxel_size

            os.unlink(flair_path)
            os.unlink(t1ce_path)

# =========================
# AFFICHAGE DES RÉSULTATS
# =========================
if st.session_state.segmentation_done:

    st.success("✅ Segmentation terminée")

    slice_id = st.slider(
        "Coupe",
        0,
        VOLUME_SLICES - 1,
        60
    )

    surface, perimetre, densite = calculer_mesures_physiques(
        st.session_state.mask[slice_id],
        st.session_state.dx_dy[0],
        st.session_state.dx_dy[1]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Surface (mm²)", f"{surface:.2f}")
    c2.metric("Périmètre (mm)", f"{perimetre:.2f}")
    c3.metric("Densité", f"{densite:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(
        cv2.resize(
            st.session_state.flair_volume[:, :, slice_id + VOLUME_START_AT],
            (IMG_SIZE, IMG_SIZE)
        ),
        cmap="gray"
    )
    ax[0].set_title("FLAIR")
    ax[0].axis("off")

    ax[1].imshow(
        cv2.resize(
            st.session_state.t1ce_volume[:, :, slice_id + VOLUME_START_AT],
            (IMG_SIZE, IMG_SIZE)
        ),
        cmap="gray"
    )
    ax[1].set_title("T1CE")
    ax[1].axis("off")

    ax[2].imshow(st.session_state.mask[slice_id], cmap="jet")
    ax[2].set_title("Segmentation")
    ax[2].axis("off")

    st.pyplot(fig)

# =========================
# RETOUR
# =========================
st.markdown("---")
if st.button("🏠 Retour à l'accueil", use_container_width=True):
    st.switch_page("app.py")

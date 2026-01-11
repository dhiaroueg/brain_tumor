import numpy as np
import nibabel as nib
import cv2
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# -----------------------------
# PARAMÈTRES
# -----------------------------
IMG_SIZE = 128
VOLUME_START_AT = 22
VOLUME_SLICES = 100
SLICE_ID = 60   # slice affiché

# -----------------------------
# Charger le modèle
# -----------------------------
model = keras.models.load_model(
    "model_x81_dcs65.h5",
    custom_objects={
        "dice_coef": None,
        "precision": None,
        "sensitivity": None,
        "specificity": None,
        "dice_coef_necrotic": None,
        "dice_coef_edema": None,
        "dice_coef_enhancing": None
    },
    compile=False
)

print("==> Modèle chargé.")

# -----------------------------
# Variables globales
# -----------------------------
flair_path = None
t1ce_path = None

# -----------------------------
# Mesures physiques (mm/mm²)
# -----------------------------
def calculer_mesures_physiques(mask_slice, dx, dy):
    """
    mask_slice : 2D (128x128) mask
    dx, dy : taille du voxel en mm
    """
    # Binaire : tumeur = 1
    binary = (mask_slice > 0).astype(np.uint8)

    # Surface en mm²
    surface_mm2 = np.sum(binary) * dx * dy

    # Contours pour périmètre
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimetre_pixels = sum(cv2.arcLength(cnt, closed=True) for cnt in contours)
    perimetre_mm = perimetre_pixels * dx  # si dx ≈ dy

    # Densité tumorale (ratio)
    surface_totale_mm2 = binary.shape[0] * binary.shape[1] * dx * dy
    densite = surface_mm2 / surface_totale_mm2 if surface_totale_mm2 > 0 else 0

    return surface_mm2, perimetre_mm, densite

# -----------------------------
# Fonctions GUI
# -----------------------------
def choisir_flair():
    global flair_path
    flair_path = filedialog.askopenfilename(
        title="Choisir le fichier FLAIR",
        filetypes=[("NIfTI files", "*.nii *.nii.gz")]
    )
    if flair_path:
        label_flair.config(text=flair_path)

def choisir_t1ce():
    global t1ce_path
    t1ce_path = filedialog.askopenfilename(
        title="Choisir le fichier T1CE",
        filetypes=[("NIfTI files", "*.nii *.nii.gz")]
    )
    if t1ce_path:
        label_t1ce.config(text=t1ce_path)

def lancer_segmentation():
    if flair_path is None or t1ce_path is None:
        messagebox.showerror(
            "Erreur",
            "Veuillez sélectionner les fichiers FLAIR et T1CE."
        )
        return

    # Charger les volumes
    flair_img = nib.load(flair_path)
    t1ce_img  = nib.load(t1ce_path)
    flair = flair_img.get_fdata()
    t1ce  = t1ce_img.get_fdata()

    # Taille des voxels en mm
    dx, dy, dz = flair_img.header.get_zooms()

    # Préparer les slices
    X = np.zeros((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)

    for i in range(VOLUME_SLICES):
        X[i, :, :, 0] = cv2.resize(
            flair[:, :, i + VOLUME_START_AT],
            (IMG_SIZE, IMG_SIZE)
        )
        X[i, :, :, 1] = cv2.resize(
            t1ce[:, :, i + VOLUME_START_AT],
            (IMG_SIZE, IMG_SIZE)
        )

    # Normalisation
    X = X / np.max(X)

    # Prédiction
    pred = model.predict(X, verbose=1)
    mask = np.argmax(pred, axis=-1)

    # Calcul des mesures physiques
    surface_mm2, perimetre_mm, densite = calculer_mesures_physiques(mask[SLICE_ID], dx, dy)

    # -----------------------------
    # Affichage
    # -----------------------------
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title("FLAIR")
    plt.imshow(
        cv2.resize(
            flair[:, :, SLICE_ID + VOLUME_START_AT],
            (IMG_SIZE, IMG_SIZE)
        ),
        cmap="gray"
    )
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmentation prédite")
    plt.imshow(mask[SLICE_ID], cmap="jet", alpha=0.7)
    plt.axis("off")

    # Texte des mesures
    plt.figtext(
        0.5, 0.02,
        f"Surface: {surface_mm2:.2f} mm² | "
        f"Périmètre: {perimetre_mm:.2f} mm | "
        f"Densité: {densite:.4f}",
        ha="center",
        fontsize=12
    )

    plt.show()

# -----------------------------
# Interface Tkinter
# -----------------------------
root = tk.Tk()
root.title("Segmentation de tumeur cérébrale")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

btn_flair = tk.Button(frame, text="Choisir FLAIR (.nii)", command=choisir_flair)
btn_flair.pack(fill="x")

label_flair = tk.Label(frame, text="Aucun fichier sélectionné", wraplength=450)
label_flair.pack()

btn_t1ce = tk.Button(frame, text="Choisir T1CE (.nii)", command=choisir_t1ce)
btn_t1ce.pack(fill="x", pady=(10, 0))

label_t1ce = tk.Label(frame, text="Aucun fichier sélectionné", wraplength=450)
label_t1ce.pack()

btn_run = tk.Button(
    frame,
    text="Lancer la segmentation",
    command=lancer_segmentation,
    bg="#4CAF50",
    fg="white"
)
btn_run.pack(fill="x", pady=15)

root.mainloop()

import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Charger le modèle sauvegardé
model = load_model("effnet.h5")
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
image_size = 150  # même taille utilisée pour l'entraînement

# Fonction pour charger et prédire l'image
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Charger et afficher l'image
        img = Image.open(file_path)
        img_resized = img.resize((image_size, image_size))
        tk_img = ImageTk.PhotoImage(img.resize((250, 250)))  # pour afficher dans l'UI
        img_label.config(image=tk_img)
        img_label.image = tk_img

        # Préparer l'image pour le modèle
        img_array = np.array(img_resized)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        pred = model.predict(img_array)
        pred_percent = pred[0] * 100
        pred_class_index = np.argmax(pred)
        pred_class = labels[pred_class_index]

        # Affichage des résultats
        result_text = f"Prédiction: {pred_class}\n\nPourcentages:\n"
        for i, label in enumerate(labels):
            result_text += f"{label}: {pred_percent[i]:.2f}%\n"
        result_label.config(text=result_text)

# Interface graphique
root = tk.Tk()
root.title("Test du modèle EfficientNetB0")
root.geometry("400x600")

btn = Button(root, text="Choisir une image", command=open_image)
btn.pack(pady=20)

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(root, text="", justify="left", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()

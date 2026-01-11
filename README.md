# Brain Tumor Classification & Segmentation

Ce projet contient deux scripts principaux :

Classification d’images IRM

Segmentation de volumes IRM

Les deux utilisent une interface graphique pour sélectionner les données et afficher les résultats.

## Prérequis

Installe Python 3.10 via conda pour éviter les incompatibilités :

conda create -n tf310 python=3.10
conda activate tf310
pip install tensorflow keras nibabel opencv-python matplotlib

1) Classification des tumeurs cérébrales

Ce script ouvre une interface graphique qui te permet :

de choisir une image IRM

de classer l’image

d’afficher la classe de tumeur prédite

Exécution :

cd classification
python classification.py

Dataset conseillé pour la classification

📌 Brain Tumor MRI Dataset (4 classes)
Ce dataset contient des milliers d’images MRI classées en :

Glioma

Meningioma

Pituitary Tumor

No Tumor
Téléchargement et description :
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri


👉 Organise les images par dossier de classe pour l’entraînement.

2) Segmentation des tumeurs cérébrales

Ce script ouvre une interface graphique pour :

sélectionner deux fichiers .nii :

FLAIR

T1CE

lancer la segmentation

afficher le résultat avec :

surface en mm²

périmètre en mm

densité tumorale (ratio)

Exécution :

cd seg
python testmodel.py

Dataset conseillé pour la segmentation

📌 Multimodal Brain Tumor Segmentation Challenge – BraTS 2020
Contient des volumes IRM multimodaux avec annotations (FLAIR, T1, T1CE, T2 et segmentations).
Pour y accéder, il faut s’inscrire et télécharger les données via le portail officiel :
https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

Ce dataset est un standard de la recherche en segmentation IRM.

Structure du projet
project/
│
├─ classification/
│   ├─ classification.py
│   └─ effnet.h5
│
├─ seg/
│   ├─ testmodel.py
│   └─ model_x81_dcs65.h5.h5
│
└─ README.md

Notes utiles
Sur les mesures physiques

Pour la segmentation, les mesures sont calculées en unités physiques (millimètres / millimètres carrés) en utilisant les informations du header NIfTI, ce qui est conforme aux standards médicaux. Cela élimine l’arbitraire des pixels et donne des valeurs exploitables cliniquement.

Dépendances Python

tensorflow

keras

nibabel

opencv-python

matplotlib

tkinter (inclus avec Python)

Conseils de bonnes pratiques

Organise tes datasets dans des dossiers clairs (train, test, etc.)

Vérifie la résolution des volumes .nii (via header.get_zooms())

Utilise les résultats de segmentation pour analyser les caractéristiques tumorales quantitatives (surface, forme, distribution)
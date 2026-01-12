markdown# 🧠 Brain Tumor Analysis System

Système complet d'analyse des tumeurs cérébrales utilisant l'intelligence artificielle pour la **segmentation 3D** et la **classification d'images médicales**.

---

## 📋 Table des Matières
- [Aperçu du Projet](#aperçu-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Architecture du Projet](#🏗️-architecture-du-projet)
- [Prérequis](#⚙️-prérequis)
- [Installation](#🚀-installation)
- [Utilisation](#💻-utilisation)
- [Structure des Fichiers](#📁-structure-des-fichiers)
- [Modèles d'IA](#🤖-modèles-dia)
- [Données d'Entraînement](#📚-données-dentraînement-des-modèles)
- [Données de Test](#📊-données-de-test)
- [Avertissements](#⚠️-avertissements)

---

## 🎯 Aperçu du Projet

Application web interactive développée avec **Streamlit**, permettant aux professionnels de santé d'analyser des images médicales cérébrales.  

Le système combine deux approches d'IA :  

- **Segmentation 3D** : analyse de volumes complets d'IRM (.nii) pour délimiter les différentes régions tumorales  
- **Classification** : identification du type de tumeur à partir d'images 2D d'IRM  

---

## ✨ Fonctionnalités

### 🧬 Segmentation 3D
- Chargement de fichiers NIfTI (FLAIR et T1CE)  
- Segmentation multi-classes : **nécrose, œdème, zone renforcée**  
- Calcul automatique des mesures :
  - Surface tumorale (mm²)  
  - Périmètre (mm)  
  - Densité tumorale  
- Visualisation interactive des coupes  
- Export des résultats en **PNG** et **CSV**  

### 🔍 Classification 2D
- Formats supportés : PNG, JPG, JPEG  
- Classification en 4 catégories :
  - Gliome  
  - Méningiome  
  - Tumeur pituitaire  
  - Aucune tumeur  
- Scores de confiance détaillés  
- Génération de rapports médicaux  
- Mode démonstration intégré  

### 👤 Interface Patient
- Formulaire d'enregistrement complet  
- Stockage sécurisé des informations médicales  
- Interface intuitive et responsive  

---

## 🏗️ Architecture du Projet
```text
BRAIN8_TUMOR
├── app.py                  # Application principale
├── pages/
│   ├── Segmentation.py     # Page de segmentation
│   └── Classification.py   # Page de classification
├── models/                 # Modèles d'IA pré-entraînés
│   ├── model_x81_dcs65.h5  # Modèle de segmentation
│   └── effnet.h5           # Modèle de classification
├── utils/                  # Fonctions utilitaires
│   └── helpers.py
├── data/                   # Données patients (générées)
├── assets/                 # Ressources statiques
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
```

---

## ⚙️ Prérequis

- Python 3.8 ou 3.9  
- 8GB RAM minimum (16GB recommandé)  
- 5GB d'espace disque libre  
- Connexion internet pour l'installation  

---

## 🚀 Installation

### 1. Cloner ou télécharger le projet
```bash
git clone https://github.com/mohemed-amine-gharbi/brain_tumor
cd BRAIN8_TUMOR
```

### 2. Créer un environnement virtuel
```bash
# Avec conda (recommandé pour Windows)
conda create -n brain_tumor python=3.9 -y
conda activate brain_tumor

# Avec venv (Linux/Mac)
python -m venv brain_tumor_env
source brain_tumor_env/bin/activate  # Linux/Mac
# Ou
brain_tumor_env\Scripts\activate  # Windows
```

### 3. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Utilisation

### Lancer l'application en ligne
✅ **Application disponible à :** https://bt-app.streamlit.app/

### Lancer l'application localement
```bash
# Naviguer vers le dossier du projet
cd BRAIN8_TUMOR

# Lancer Streamlit
streamlit run app.py
```

L'application sera accessible à l'adresse : **http://localhost:8501**

### Interface Utilisateur

**Page d'accueil (app.py) :**
- Remplir le formulaire patient  
- Choisir entre segmentation ou classification  

**Segmentation 3D (pages/Segmentation.py) :**
```
Étapes :
1. Télécharger fichier FLAIR (.nii)
2. Télécharger fichier T1CE (.nii)
3. Cliquer sur "Lancer la segmentation"
4. Visualiser les résultats
5. Exporter les mesures
```

**Classification (pages/Classification.py) :**
```
Étapes :
1. Télécharger une image d'IRM (.png, .jpg, .jpeg)
2. Cliquer sur "Analyser l'image"
3. Consulter les résultats
4. Télécharger le rapport
```

---

## 📁 Structure des Fichiers

### Fichiers Principaux

| Fichier | Description |
|---------|-------------|
| `app.py` | Application principale avec formulaire patient |
| `pages/Segmentation.py` | Segmentation 3D des volumes NIfTI |
| `pages/Classification.py` | Classification des images 2D |
| `requirements.txt` | Liste des dépendances Python |

### Dossiers

| Dossier | Contenu |
|---------|---------|
| `models/` | Modèles d'IA pré-entraînés |
| `data/` | Données patients (auto-généré) |
| `utils/` | Fonctions utilitaires |

---

## 🤖 Modèles d'IA

### Modèle de Segmentation

- **Format :** .h5 (Keras)  
- **Architecture :** U-Net ou similaire  
- **Entrée :** 128×128×2 (FLAIR + T1CE)  
- **Sortie :** 3 classes (nécrose, œdème, renforcée)  
- **Performance :** Dice coefficient ≈ 0.81  

### Modèle de Classification

- **Format :** .h5 (Keras)  
- **Architecture :** EfficientNetB0  
- **Entrée :** 150×150×3 (RGB)  
- **Sortie :** 4 classes  
- **Accuracy :** > 90% (sur données test)  

---

## 📚 Données d'Entraînement des Modèles

### 🧬 Données d'Entraînement – Segmentation 3D

Le modèle de segmentation 3D a été entraîné à l'aide du jeu de données **BraTS (Brain Tumor Segmentation)**, qui contient des volumes IRM avec annotations multi-classes pour les différentes régions tumorales.

**Dataset utilisé :**  
🔗 https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

**Description :**
- Modalités d'IRM : FLAIR, T1, T1CE, T2  
- Anatomie annotée en plusieurs classes (nécrose, œdème, zone active, etc.)  
- Format : volumes NIfTI (.nii/.nii.gz)  
- Standard de référence pour l'entraînement de modèles de segmentation 3D  

### 🔍 Données d'Entraînement – Classification 2D

Le modèle de classification est entraîné sur un jeu d'images IRM étiquetées en différentes catégories de tumeurs cérébrales.

**Dataset utilisé :**  
🔗 https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

**Classes :**
- Gliome  
- Méningiome  
- Tumeur pituitaire  
- Pas de tumeur  

**Format :** images 2D (.jpg, .png, .jpeg)

### 📝 Notes additionnelles

- Les datasets sont utilisés à des fins de recherche et d'entraînement uniquement  
- Veuillez consulter les licences sur les plateformes respectives (Kaggle) avant de redistribuer les données  

---

## 📊 Données de Test

### Fichiers NIfTI de démonstration

Pour tester la segmentation, vous pouvez utiliser des données publiques :

- **BraTS Dataset :** https://www.med.upenn.edu/cbica/brats/  
- **Sample NIfTI :** Utiliser des données d'entraînement du challenge BraTS  

### Images de test pour la classification

Des exemples d'images sont disponibles dans :

- **Kaggle Brain Tumor Dataset :** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset  
- **Figshare :** Rechercher "brain tumor MRI dataset"  

---

## ⚠️ Avertissements

- Ce système est destiné à un usage de **recherche et démonstration**  
- Les résultats ne doivent **pas être utilisés pour un diagnostic médical officiel**  
- Toujours consulter un professionnel de santé qualifié pour l'interprétation clinique

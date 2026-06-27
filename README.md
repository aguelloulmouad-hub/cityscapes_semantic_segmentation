# Projet Deep Learning Avancé – P-02 : Segmentation Sémantique de Scènes Urbaines avec U-Net et DeepLabV3+

Ce dépôt contient le notebook Jupyter complet du projet **P-02 – Vision par Ordinateur** du module **Deep Learning Avancé** (Master AIDC, FST Béni Mellal, Université Sultan Moulay Slimane).

L'objectif est de développer un système de **segmentation sémantique** capable d'effectuer une classification **pixel par pixel** de scènes urbaines à partir du dataset **Cityscapes**. Deux architectures modernes, **U-Net** et **DeepLabV3+**, sont entraînées, comparées et évaluées à l'aide de métriques adaptées à la segmentation. Le meilleur modèle est ensuite utilisé pour segmenter une vidéo de conduite.

---

## 📋 Contenu

```
cityscapes-semantic-segmentation/
│
├── p02-segmentation-semantique-cityscapes.ipynb
├── README.md
│
└── video_demo/
    ├── segment_video.py
    └── setup_and_run.py
```

- **p02-segmentation-semantique-cityscapes.ipynb** : notebook principal contenant l'ensemble du pipeline d'entraînement, d'évaluation et de comparaison des modèles.
- **video_demo/segment_video.py** : script permettant d'appliquer le meilleur modèle entraîné à une vidéo.
- **video_demo/setup_and_run.py** : script d'installation des dépendances et de lancement de la démonstration.

---

## 🚀 Principales fonctionnalités

- Prétraitement du dataset **Cityscapes**.
- Visualisation des images et des masques de segmentation.
- Analyse de la distribution des classes.
- Data augmentation pour améliorer la généralisation.
- Entraînement de deux architectures :
  - **U-Net (ResNet50 Encoder)**
  - **DeepLabV3+ (ResNet50 Encoder)**
- Sauvegarde automatique du meilleur modèle.
- Comparaison des performances des architectures.
- Évaluation quantitative et qualitative.
- Segmentation d'une vidéo avec le meilleur modèle (**DeepLabV3+**).

---

## 📊 Évaluation

Les modèles sont évalués à l'aide des métriques suivantes :

- Intersection over Union (IoU)
- Mean Intersection over Union (mIoU)
- Dice Score
- Mean Dice Score (mDice)
- Accuracy par pixel
- Courbes d'entraînement (Loss et métriques)

Une comparaison finale met en évidence les performances respectives de **U-Net** et **DeepLabV3+**, permettant de sélectionner le modèle le plus performant.

---

## 🛠 Technologies utilisées

- Python 3.10+
- PyTorch
- segmentation-models-pytorch
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- Scikit-learn
- Weights & Biases (W&B)

---

## 📂 Dataset

Le projet utilise le dataset **Cityscapes**, une référence pour la segmentation sémantique de scènes urbaines.

Le notebook est basé sur la version Kaggle :

- **Cityscapes Image Pairs**
  - 2 975 images d'entraînement
  - 500 images de validation

---

## ▶️ Exécution

Le notebook a été conçu pour être exécuté sur **Kaggle** avec **2 GPU Tesla T4**, mais peut également être utilisé sur Google Colab ou sur toute machine équipée d'un GPU compatible CUDA.

Les principales étapes sont :

1. Installation des dépendances.
2. Chargement et préparation du dataset.
3. Data augmentation.
4. Création des DataLoaders.
5. Entraînement de U-Net.
6. Entraînement de DeepLabV3+.
7. Évaluation des modèles.
8. Comparaison des performances.
9. Sauvegarde du meilleur modèle.
10. Démonstration sur une vidéo.

---

## 🎥 Démonstration vidéo

Le dossier **video_demo/** permet de tester directement le meilleur modèle entraîné (**DeepLabV3+**) sur une vidéo.

Exécution :

```bash
python video_demo/setup_and_run.py
```

ou directement

```bash
python video_demo/segment_video.py
```

Le script :

- charge le modèle sauvegardé ;
- lit la vidéo image par image ;
- applique la segmentation sémantique ;
- génère une vidéo segmentée.

---

## 📌 Résultats

Le projet compare les performances de **U-Net** et **DeepLabV3+** sur le dataset Cityscapes.

Les résultats incluent notamment :

- évolution des pertes d'entraînement et de validation ;
- comparaison des métriques IoU et Dice ;
- visualisation qualitative des prédictions ;
- comparaison finale des deux architectures.

Le meilleur modèle obtenu est **DeepLabV3+ (ResNet50)**, utilisé pour la démonstration vidéo.

---

## 📝 Licence & crédits

Projet réalisé dans le cadre du module **Deep Learning Avancé**  
**Master Artificial Intelligence & Digital Computing (AIDC)**  
**Faculté des Sciences et Techniques de Béni Mellal**  
**Université Sultan Moulay Slimane**

Dataset :

- Cityscapes Dataset

Frameworks :

- PyTorch
- segmentation-models-pytorch

---

*Notebook créé et exécuté par **Aguelloul Mouad** et **Dahibi Wissal** – Master AIDC 2025–2026.*

# Emotion Recognition from Facial Expressions

🎯 Ce projet utilise un modèle de réseau de neurones convolutionnels (CNN) pour la reconnaissance des émotions à partir des expressions faciales. Le modèle est entraîné sur le dataset **FER-2013** et utilise OpenCV pour capturer des images à partir de la webcam et détecter les émotions en temps réel.

## Prérequis ⚙️

Avant de commencer, assurez-vous d'avoir les outils suivants installés :

- 🐍 Python 3.x
- 🖼️ OpenCV
- 🤖 TensorFlow
- 🤖 Keras
- 📊 Pandas
- 📈 Matplotlib
- 🔍 Scikit-learn
- 🔢 Numpy

Vous pouvez installer les dépendances nécessaires avec la commande suivante :

```bash
pip install opencv-python tensorflow pandas matplotlib scikit-learn numpy
```
## Structure du projet 📁
### Le projet est divisé en deux parties principales :

#### 1. Entraînement du modèle 
Dans cette partie, nous entraînons un modèle de reconnaissance des émotions basé sur les expressions faciales. Le modèle est entraîné sur les données du fichier fer2013.csv, qui contient 35 000 images de visages étiquetées avec 7 émotions différentes.

#### 2. Détection des émotions en temps réel via la webcam
Dans cette partie, nous chargeons le modèle préalablement entraîné et utilisons OpenCV pour détecter les visages dans un flux vidéo provenant de la webcam. Le modèle prédit l'émotion à partir des expressions faciales détectées et l'affiche en temps réel.

## Fonctionnement du projet 🧠
### Entraînement du modèle 🏋️ :

Le modèle est un réseau de neurones convolutionnels qui prend en entrée des images de visages (48x48 pixels en niveaux de gris).
Le modèle est entraîné sur le dataset FER-2013 pour reconnaître 7 émotions différentes :

😡 Colère
🤢 Dégoût
😨 Peur
😄 Joie
😢 Tristesse
😲 Surprise
😐 Neutre

### Détection en temps réel via la webcam 📸 :

Une fois le modèle entraîné, vous pouvez l'utiliser pour prédire les émotions à partir des visages détectés dans un flux vidéo en temps réel.
Le modèle prédit l'émotion du visage et affiche le label de l'émotion sur l'image en temps réel.

## Comment exécuter le projet 🚀
#### Entraîner le modèle 🔨 :

Exécutez le script d'entraînement pour entraîner le modèle sur le dataset FER-2013 :
```bash
python train_model.py
```
Cela entraînera le modèle et le sauvegardera sous le nom modele_emotion.h5.

#### Détection des émotions via la webcam 🖥️ :

Une fois le modèle entraîné, vous pouvez lancer le script de détection en temps réel :
```bash
python detect_emotions.py
```
Le programme ouvrira la webcam et affichera l'émotion prédite pour chaque visage détecté.

## Résultats attendus 📊
Le programme de détection des émotions via la webcam devrait être capable de reconnaître les émotions à partir des expressions faciales.
Le modèle peut être amélioré avec plus de données ou des techniques d'augmentation de données pour augmenter la précision.

## Contributions 🤝
Les contributions à ce projet sont les bienvenues ! Si vous avez des suggestions ou des améliorations à apporter, n'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence 📜
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

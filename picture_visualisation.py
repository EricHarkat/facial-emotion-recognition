import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

"""
    1 - Prétraitement des données : Chargement des images, mise en forme, normalisation et encodage des labels (emotions).
    2 - Séparation des ensembles : Division en train, validation et test.
    3 - Construction du modèle CNN : 3 couches de convolution, pooling, flattening et fully connected.
    4 - Optimisation : Utilisation de EarlyStopping et ReduceLROnPlateau pour éviter l'overfitting.
    5 - Entraînement et évaluation : Entraînement sur 50 epochs et sauvegarde du modèle entraîné.
"""
 
# Charger le fichier CSV ( 35 000 image de visages sous forme de texte )
data = pd.read_csv('./fer2013.csv')
 
# Extraire les labels (émotions) 
labels = data['emotion'].values
 
# Extraire les pixels et les convertir en images
def process_pixels(pixels):
    # Les pixels sont une chaîne de caractères, il faut les convertir en liste d'entiers
    pixels = [int(p) for p in pixels.split()]
    # Convertir la liste en tableau numpy et redimensionner
    image = np.array(pixels).reshape(48, 48)
    # Redimensionner en format approprié (ici, on garde 48x48)
    image = cv2.resize(image, (48, 48))
    return image
 
# Appliquer le prétraitement sur toutes les images
images = np.array([process_pixels(pixels) for pixels in data['pixels'].values])
 
'''
    Ajouter une dimension supplémentaire pour les canaux de couleur pour etre compatible avec le model de deep learning
    (en niveaux de gris, il n'y a qu'un seul canal)
'''
images = images.reshape(-1, 48, 48, 1)
 
# Normaliser les valeurs des pixels entre 0 et 1
images = images / 255.0
 
"""
    Extraire les labels et les convertir en format catégorique (one-hot encoding)
    les emotions sont converties en categories numeriques
    ex : joie → [1, 0, 0, 0, 0, 0, 0], Tristesse → [0, 1, 0, 0, 0, 0, 0], etc
    FER-2013 a 7 émotions
"""
labels = to_categorical(labels, num_classes=7)
"""
Diviser les données en ensembles de formation et de test
On divise les images en train (80%), validation (10%), et test (10%)
Train (80%) : pour entraîner le modèle, Validation (10%) : pour ajuster les paramètres du modèle, Test (10%) : pour évaluer la performance finale du modèle."
"""
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
 
# Diviser à nouveau l'ensemble d'entraînement pour créer un ensemble de validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
 
# Afficher les dimensions des ensembles de données
print(f"Dimensions des images d'entraînement : {X_train.shape}")
print(f"Dimensions des labels d'entraînement : {y_train.shape}")
print(f"Dimensions des images de validation : {X_val.shape}")
print(f"Dimensions des labels de validation : {y_val.shape}")
print(f"Dimensions des images de test : {X_test.shape}")
print(f"Dimensions des labels de test : {y_test.shape}")
 
# Pour vérifier si le chargement des images fonctionne bien :
plt.imshow(X_train[0].reshape(48, 48), cmap='gray')
plt.title(np.argmax(y_train[0]))  # titre avec l'émotion de l'étiquette
plt.show()
 
"""
    Le programme construit un réseau de neurones convolutionnels pour analyser les images
    On utilise 3 couches de convolution ici (Conv2D) et un filtre de pooling 2 x 2
"""
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)), # 48 x 48 pixels et 1 canal car gris
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Corrected dropout rate
    Dense(7, activation='softmax')  # 7 types d'émotion
])
 
# Résumé du modèle
model.summary()
 
"""
    Utiliser EarlyStopping et ReduceLROnPlateau pour améliorer l'entraînement
    EarlyStopping arrête l'entraînement si la performance ne s'améliore plus après 5 epochs.
    ReduceLROnPlateau réduit le taux d’apprentissage si la validation stagne.
"""
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
 
"""
    Le modèle est compilé avec l'algorithme Adam et la fonction de perte categorical_crossentropy (adaptée aux classifications multi-classes).
    cette etapes sert a dire au model comment il va apprendre et comment il va mesurer sa performance
    et compile permet de definir 3 elements essentiel : optimizer, loss, metrics
"""
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# On entraîne le modèle sur 50 epochs avec des mini-batches de 64 images :
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr])
 
# Évaluer le modèle : Après l'entraînement, le programme teste la précision du modèle sur les données de test :
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Le model est enregistre
model.save("modele_emotion.h5")
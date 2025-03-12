import cv2
import numpy as np
from tensorflow.keras.models import load_model

"""
    Détection des visages en temps réel avec Haar Cascade.
    Extraction et prétraitement du visage détecté (redimensionnement, normalisation).
    Classification de l’émotion avec un modèle CNN pré-entraîné.
    Affichage du résultat sur l’image avec OpenCV.
    Fermeture propre du programme sur pression de la touche "q".
"""

# Charger le modèle entraîné
model = load_model("modele_emotion.h5")  # Assure-toi que le modèle est bien enregistré sous ce nom

# Dictionnaire des émotions
emotion_labels = ["Colere", "Degout", "Peur", "Joie", "Tristesse", "Surprise", "Neutre"]

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

# Charger un détecteur de visage (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Lire l'image de la webcam; ret est un booléen indiquant si la capture a réussi.
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris ; (nécessaire pour l’algorithme de détection de visage).
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecte les visages dans l’image.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) # scaleFactor=1.3 : réduit la taille de l’image pour améliorer la détection.

    for (x, y, w, h) in faces:
        # Extraire la région du visage
        face = gray[y:y + h, x:x + w]

        # Redimensionner l'image en 48x48 pixels
        face_resized = cv2.resize(face, (48, 48))

        # Normaliser les pixels
        face_resized = face_resized / 255.0

        # Ajouter une dimension pour correspondre à l'entrée du modèle (48,48,1)
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = np.expand_dims(face_resized, axis=-1)

        # Faire une prédiction avec le modèle
        prediction = model.predict(face_resized)
        emotion_index = np.argmax(prediction)  # Récupérer l’émotion prédite

        # Affiche le nom de l’émotion prédite au-dessus du visage détecté.
        cv2.putText(frame, emotion_labels[emotion_index], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Dessine un rectangle vert autour du visage détecté.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow("Détection des émotions", frame)

    # Attente d’une entrée clavier (q) pour quitter la boucle et fermer la webcam.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer la webcam et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()

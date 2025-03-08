import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle entraîné
model = load_model("modele_emotion.h5")  # Assure-toi que le modèle est bien enregistré sous ce nom

# Dictionnaire des émotions
emotion_labels = ["Colere", "Degout", "Peur", "Joie", "Tristesse", "Surprise", "Neutre"]

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

# Charger un détecteur de visage (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Lire l'image de la webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

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

        # Ajouter le label sur l'image
        cv2.putText(frame, emotion_labels[emotion_index], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow("Détection des émotions", frame)

    # Quitter avec la touche "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer la webcam et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()

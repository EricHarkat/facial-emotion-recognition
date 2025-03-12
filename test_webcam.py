import cv2

"""
    Pour tester si la webcam est detecter
    Lancer la commande python test_webcam.py
"""

cap = cv2.VideoCapture(0
)  # Essaie aussi avec 1 ou 2 si 0 ne fonctionne pas

if not cap.isOpened():
    print("Erreur : Webcam non détectée !")
else:
    print("Webcam détectée avec succès !")
cap.release()

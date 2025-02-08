#!/usr/bin/env python3

import cv2

video_path = "extrait.mkv"
cap = cv2.VideoCapture(video_path)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Extraction de la ROI du logo
    coord_x = 1740
    coord_y = 65
    largeur = 85
    hauteur = 95
    x, y, w, h = roi = (coord_x, coord_y, largeur, hauteur)  # à définir
    logo_region = frame[y:y+h, x:x+w]
    
    # Sauvegarder l'image de la ROI
    cv2.imwrite(f"logos-true-pos/frame_{count}.png", logo_region)
    count += 1

cap.release()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from typing import List, Tuple

# -------------------------------------------------------------
# PARAMÈTRES GLOBAUX
# -------------------------------------------------------------
LOGO_COORDS = (1740, 65, 85, 95)   # (x, y, largeur, hauteur)
LOGO_THRESHOLD = 0.8              # Seuil pour cv2.matchTemplate

MIN_AD_DURATION_SECONDS = 180     # Durée minimale d'une pub en secondes (ici 3 minutes)

# Paramètres pour l'extraction des keyframes
KEYFRAME_SKIP = 10                # On analyse une frame sur 10 pour accélérer l'extraction
KEYFRAME_DIFF_THRESHOLD = 30.0    # Seuil de différence moyenne pour retenir une keyframe

# -------------------------------------------------------------
# FONCTIONS DE CHARGEMENT ET DE DÉTECTION DU LOGO
# -------------------------------------------------------------
def load_logo_images(logo_folder: str) -> List[np.ndarray]:
    """
    Charge toutes les images (formats png, jpg, jpeg, bmp) du dossier 'logo_folder'.
    """
    logos = []
    for file_name in os.listdir(logo_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(logo_folder, file_name)
            logo_img = cv2.imread(path)
            if logo_img is not None:
                logos.append(logo_img)
    return logos

def is_logo_present(frame: np.ndarray, logos: List[np.ndarray]) -> bool:
    """
    Vérifie si, dans la zone définie par LOGO_COORDS, le logo est présent dans 'frame'.
    Pour chaque image de référence, on applique cv2.matchTemplate et on compare le score.
    """
    if frame is None:
        return False

    x, y, w, h = LOGO_COORDS
    frame_h, frame_w = frame.shape[:2]
    # Vérification que la zone d'analyse est bien dans la frame
    if x + w > frame_w or y + h > frame_h:
        return False

    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    for logo in logos:
        # Si la taille du logo n'est pas exactement celle de la ROI, on redimensionne
        if logo.shape[1] != w or logo.shape[0] != h:
            logo_resized = cv2.resize(logo, (w, h))
        else:
            logo_resized = logo
        logo_gray = cv2.cvtColor(logo_resized, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(roi_gray, logo_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val >= LOGO_THRESHOLD:
            return True
    return False

def frames_to_timecode(frame_index: int, fps: float) -> str:
    """
    Convertit un numéro de frame en timecode (hh:mm:ss).
    """
    total_seconds = frame_index / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# -------------------------------------------------------------
# EXTRACTION DES KEYFRAMES
# -------------------------------------------------------------
def extract_keyframes(video_path: str, skip: int = KEYFRAME_SKIP, diff_threshold: float = KEYFRAME_DIFF_THRESHOLD) -> List[Tuple[int, np.ndarray]]:
    """
    Parcourt la vidéo et extrait les keyframes.
    On ne traite qu'une frame sur 'skip'. La première frame est toujours retenue.
    Si la différence moyenne (en niveaux de gris) entre la frame courante et la dernière keyframe
    dépasse 'diff_threshold', la frame est ajoutée à la liste des keyframes.
    
    Retourne une liste de tuples (frame_index, frame).
    """
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    last_keyframe_gray = None
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # On traite seulement une frame sur 'skip'
        if frame_index % skip != 0:
            frame_index += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_keyframe_gray is None:
            last_keyframe_gray = gray
            keyframes.append((frame_index, frame))
        else:
            diff = cv2.absdiff(gray, last_keyframe_gray)
            mean_diff = diff.mean()
            if mean_diff > diff_threshold:
                keyframes.append((frame_index, frame))
                last_keyframe_gray = gray
        frame_index += 1

    cap.release()
    return keyframes

# -------------------------------------------------------------
# DÉTECTION DES SEGMENTS PUBLICITAIRES PAR KEYFRAMES
# -------------------------------------------------------------
def process_keyframes(keyframes: List[Tuple[int, np.ndarray]], logos: List[np.ndarray],
                      fps: float, min_ad_duration_frames: int) -> List[Tuple[int, int]]:
    """
    Parcourt la liste des keyframes et détecte les segments publicitaires.
    On considère qu'un segment publicitaire correspond à une suite contiguë de keyframes
    où le logo n'est pas présent.
    
    Retourne une liste de tuples (start_frame, end_frame) représentant les bornes (indices dans la vidéo).
    """
    ad_segments = []
    n = len(keyframes)
    i = 0
    while i < n:
        key_idx, keyframe = keyframes[i]
        if not is_logo_present(keyframe, logos):
            # Début potentiel d'un segment pub
            ad_start = key_idx
            j = i + 1
            while j < n and not is_logo_present(keyframes[j][1], logos):
                j += 1
            # La dernière keyframe consécutive sans logo
            ad_end = keyframes[j - 1][0] if j - 1 >= i else key_idx
            # Vérifier la durée du segment (en nombre de frames)
            if (ad_end - ad_start) >= min_ad_duration_frames:
                ad_segments.append((ad_start, ad_end))
            i = j
        else:
            i += 1
    return ad_segments

# -------------------------------------------------------------
# FONCTION PRINCIPALE
# -------------------------------------------------------------
def main(video_path: str, logo_folder: str):
    # 1) Chargement des logos
    logos = load_logo_images(logo_folder)
    if not logos:
        print("[ERREUR] Aucun logo n'a été chargé depuis le dossier:", logo_folder)
        return

    # 2) Récupérer quelques infos sur la vidéo (fps et total des frames)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    min_ad_duration_frames = int(fps * MIN_AD_DURATION_SECONDS)

    print(f"Vidéo : {video_path}")
    print(f" - FPS : {fps}")
    print(f" - Total frames : {total_frames}")
    print(f" - Durée minimale d'une pub (frames) : {min_ad_duration_frames}")

    # 3) Extraction des keyframes
    print("Extraction des keyframes...")
    keyframes = extract_keyframes(video_path)
    print(f"-> {len(keyframes)} keyframes extraites.")

    # 4) Analyse des keyframes pour détecter les segments publicitaires
    print("Analyse des keyframes pour détecter les pubs...")
    ad_segments = process_keyframes(keyframes, logos, fps, min_ad_duration_frames)

    if not ad_segments:
        print("Aucun segment publicitaire détecté.")
        return

    # 5) Affichage des segments publicitaires détectés
    print("\nSegments publicitaires détectés :")
    for start_frame, end_frame in ad_segments:
        start_tc = frames_to_timecode(start_frame, fps)
        end_tc = frames_to_timecode(end_frame, fps)
        duration = (end_frame - start_frame) / fps
        print(f" - Pub de {start_tc} à {end_tc} (Durée : {duration:.1f} s)")

# -------------------------------------------------------------
# POINT D'ENTRÉE
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage : python detect-ads-by-keyframes.py <video_path> <logo_folder>")
        sys.exit(1)
    
    video_path_arg = sys.argv[1]
    logo_folder_arg = sys.argv[2]
    main(video_path_arg, logo_folder_arg)



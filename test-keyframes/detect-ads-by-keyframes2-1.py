#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import subprocess
import numpy as np
from typing import List, Tuple

# -------------------------------------------------------------
# CONSTANTES ET PARAMÈTRES
# -------------------------------------------------------------
LOGO_COORDS = (1740, 65, 85, 95)   # (x, y, largeur, hauteur)
LOGO_THRESHOLD = 0.8              # Seuil pour matchTemplate

MIN_AD_DURATION_SECONDS = 180     # Durée minimale d'une pub (3 minutes) en secondes

# -------------------------------------------------------------
# FONCTIONS UTILITAIRES
# -------------------------------------------------------------
def get_keyframe_timestamps(video_path: str) -> List[float]:
    """
    Utilise ffprobe pour récupérer la liste des timestamps (en secondes)
    de toutes les keyframes (frames de type I) de la vidéo.
    """
    command = [
        "ffprobe",
        "-select_streams", "v:0",
        "-skip_frame", "nokey",
        "-show_frames",
        "-show_entries", "frame=pkt_pts_time",
        "-of", "csv=print_section=0",
        "-i", video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        # Conversion en float
        keyframe_timestamps = [float(line) for line in lines if line.strip() != ""]
        return sorted(keyframe_timestamps)
    except subprocess.CalledProcessError:
        print("[ERREUR] Échec lors de l'exécution de ffprobe pour extraire les keyframes.")
        return []

def load_logo_images(logo_folder: str) -> List[np.ndarray]:
    """
    Charge toutes les images du dossier 'logo_folder' (formats png, jpg, jpeg, bmp)
    qui serviront de modèles de comparaison.
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
    Vérifie si, dans la zone définie par LOGO_COORDS, le logo est présent.
    Pour chaque image de référence, on applique cv2.matchTemplate et on compare
    le score obtenu au seuil LOGO_THRESHOLD.
    """
    if frame is None:
        return False

    x, y, w, h = LOGO_COORDS
    frame_h, frame_w = frame.shape[:2]
    # Vérification que la zone demandée se trouve bien dans la frame
    if x + w > frame_w or y + h > frame_h:
        return False

    roi = frame[y:y+h, x:x+w]
    # Conversion en niveaux de gris pour accélérer la comparaison
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    for logo in logos:
        # Redimensionnement éventuel du logo pour qu'il corresponde à la ROI
        if logo.shape[1] != w or logo.shape[0] != h:
            logo_resized = cv2.resize(logo, (w, h))
        else:
            logo_resized = logo
        logo_gray = cv2.cvtColor(logo_resized, cv2.COLOR_BGR2GRAY)

        # Application de matchTemplate
        result = cv2.matchTemplate(roi_gray, logo_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val >= LOGO_THRESHOLD:
            return True

    return False

def frames_to_timecode(frame_index: int, fps: float) -> str:
    """
    Convertit un numéro de frame en timecode au format hh:mm:ss.
    """
    total_seconds = frame_index / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# -------------------------------------------------------------
# FONCTIONS DE RECHERCHE DE PUB (SUR LES KEYFRAMES)
# -------------------------------------------------------------
def find_ad_segment_boundaries(
    cap: cv2.VideoCapture,
    logos: List[np.ndarray],
    keyframes: List[float],
    idx_missing_logo: int,
    fps: float,
    total_duration: float
) -> Tuple[float, float]:
    """
    Étant donné qu'à l'index `idx_missing_logo` dans la liste `keyframes` le logo est absent,
    on cherche :
      - La keyframe juste AVANT qui serait la dernière avec logo présent
      - La keyframe juste APRÈS qui serait la première avec logo présent (fin de la pub)
    
    On renvoie un tuple (start_time_sec, end_time_sec).
    """
    # 1) Trouver la dernière keyframe AVANT idx_missing_logo où le logo était présent
    start_key_idx = idx_missing_logo
    while start_key_idx >= 0:
        t = keyframes[start_key_idx]
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        if is_logo_present(frame, logos):
            # On s'arrête ici : c'est la dernière frame où le logo était présent
            break
        start_key_idx -= 1
    # start_key_idx peut être négatif si on n'a trouvé aucune frame "présente"

    # 2) Trouver la première keyframe APRÈS idx_missing_logo où le logo réapparaît
    end_key_idx = idx_missing_logo
    while end_key_idx < len(keyframes):
        t = keyframes[end_key_idx]
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        if is_logo_present(frame, logos):
            # On s'arrête : c'est la première frame où le logo réapparaît
            break
        end_key_idx += 1

    # Calcul des temps de pub_start / pub_end en seconde
    if start_key_idx < 0:
        # Logo absent dès le début
        pub_start_sec = 0.0
    else:
        pub_start_sec = keyframes[start_key_idx]
    if end_key_idx >= len(keyframes):
        # Logo absent jusqu'à la fin
        pub_end_sec = total_duration
    else:
        pub_end_sec = keyframes[end_key_idx]

    return (pub_start_sec, pub_end_sec)

def detect_ads_by_keyframes(
    cap: cv2.VideoCapture,
    logos: List[np.ndarray],
    keyframes: List[float],
    fps: float,
    total_duration: float
) -> List[Tuple[float, float]]:
    """
    Parcourt la liste des keyframes, détecte celles où le logo est absent,
    et essaye de déterminer des segments publicitaires (start_sec, end_sec).
    """
    ad_segments = []
    min_ad_duration_frames = int(MIN_AD_DURATION_SECONDS * fps)

    idx = 0
    while idx < len(keyframes):
        t_sec = keyframes[idx]
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        if not is_logo_present(frame, logos):
            # Logo absent → potentielle pub
            pub_start_sec, pub_end_sec = find_ad_segment_boundaries(
                cap, logos, keyframes, idx, fps, total_duration
            )
            pub_start_frame = int(pub_start_sec * fps)
            pub_end_frame   = int(pub_end_sec   * fps)

            if pub_end_frame > pub_start_frame and \
               (pub_end_frame - pub_start_frame) >= min_ad_duration_frames:
                ad_segments.append((pub_start_sec, pub_end_sec))
                # On avance idx pour ne pas retomber sur la même séquence
                while idx < len(keyframes) and keyframes[idx] < pub_end_sec:
                    idx += 1
                continue

        idx += 1

    # Fusion de segments qui se chevauchent
    ad_segments.sort()
    merged_segments = []
    for seg in ad_segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            last_seg = merged_segments[-1]
            if seg[0] <= last_seg[1]:
                merged_segments[-1] = (last_seg[0], max(last_seg[1], seg[1]))
            else:
                merged_segments.append(seg)

    return merged_segments

# -------------------------------------------------------------
# FONCTION PRINCIPALE
# -------------------------------------------------------------
def main(video_path: str, logo_folder: str):
    """
    1. Récupère la liste des timestamps (en sec) de toutes les keyframes de la vidéo.
    2. Parcourt uniquement ces keyframes pour détecter l’absence de logo.
    3. Détermine des segments publicitaires potentiels.
    4. Affiche les résultats (timecodes et durées).
    """
    # 1) Chargement des images du logo
    logos = load_logo_images(logo_folder)
    if not logos:
        print("[ERREUR] Aucun logo n'a été chargé. Vérifiez le dossier:", logo_folder)
        return

    # 2) Ouverture de la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo : {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Si fps = 0 (cas rare si la vidéo pose un souci), on évite la division par zéro
    if fps <= 0:
        print("[ERREUR] Le FPS détecté est 0 ou négatif, impossible de continuer.")
        cap.release()
        return

    # Calcul de la durée via (total_frames / fps)
    total_duration = total_frames / fps

    # Récupération de la liste des timestamps (en secondes) des keyframes
    keyframes = get_keyframe_timestamps(video_path)
    if not keyframes:
        print("[AVERTISSEMENT] Aucune keyframe extraite ou échec de ffprobe. "
              "Impossible de poursuivre la détection.")
        cap.release()
        return

    print(f"Vidéo : {video_path}")
    print(f"- FPS = {fps}")
    print(f"- Nombre total de frames = {total_frames}")
    print(f"- Durée (secondes) = {total_duration:.1f}")
    print(f"- Nombre de keyframes = {len(keyframes)}")

    # 3) Détection des pubs via keyframes
    ad_segments = detect_ads_by_keyframes(cap, logos, keyframes, fps, total_duration)
    cap.release()

    # 4) Affichage des segments publicitaires trouvés
    if not ad_segments:
        print("Aucune publicité détectée après analyse (keyframes).")
        return

    print("\nSegments publicitaires détectés (basés sur keyframes) :")
    for (start_sec, end_sec) in ad_segments:
        start_frame = int(start_sec * fps)
        end_frame   = int(end_sec   * fps)
        start_tc    = frames_to_timecode(start_frame, fps)
        end_tc      = frames_to_timecode(end_frame,   fps)
        duration    = end_sec - start_sec
        print(f" - Pub de {start_tc} à {end_tc} (Durée: {duration:.1f} s)")

# -------------------------------------------------------------
# POINT D'ENTRÉE
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python detect-ads-keyframes.py <video_path> <logo_folder>")
        sys.exit(1)

    video_path_arg = sys.argv[1]
    logo_folder_arg = sys.argv[2]
    main(video_path_arg, logo_folder_arg)



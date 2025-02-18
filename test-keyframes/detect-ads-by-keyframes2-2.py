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
def get_keyframes(video_path: str) -> List[int]:
    """
    Récupère la liste des numéros de frame des keyframes avec ffprobe
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=key_frame,pkt_pts_time',
        '-of', 'csv=nokey=1',
        video_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    keyframes = []
    frame_number = 0
    
    for line in result.stdout.split('\n'):
        if line:
            key_flag, pts_time = line.split(',')
            if key_flag == '1':
                keyframes.append(frame_number)
            frame_number += 1
    
    return keyframes

# -------------------------------------------------------------
# FONCTIONS MODIFIÉES POUR UTILISER LES KEYFRAMES
# -------------------------------------------------------------
def process_keyframes(cap: cv2.VideoCapture, logos: List[np.ndarray], 
                     keyframes: List[int], fps: float) -> List[Tuple[int, int]]:
    """
    Analyse les keyframes pour détecter les segments sans logo
    """
    ad_segments = []
    min_ad_frames = int(fps * MIN_AD_DURATION_SECONDS)
    
    # Analyse initiale des keyframes
    logo_presence = []
    for kf in keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf)
        ret, frame = cap.read()
        if ret:
            logo_presence.append((kf, is_logo_present(frame, logos)))
    
    # Détection des transitions
    prev_present = True
    current_start = -1
    
    for i, (kf, present) in enumerate(logo_presence):
        if not present and prev_present:
            current_start = kf
        elif present and not prev_present and current_start != -1:
            current_end = kf
            if (current_end - current_start) >= min_ad_frames:
                ad_segments.append((current_start, current_end))
            current_start = -1
        prev_present = present
    
    return ad_segments

def refine_segment(cap: cv2.VideoCapture, logos: List[np.ndarray],
                  start_kf: int, end_kf: int) -> Tuple[int, int]:
    """
    Affine les limites d'un segment entre deux keyframes
    """
    # Affinage du début
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_kf)
    ret, frame = cap.read()
    if not ret or is_logo_present(frame, logos):
        return (0, 0)
    
    # Recherche arrière pour la transition présent->absent
    start = start_kf
    while start > 0:
        start -= 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame = cap.read()
        if ret and is_logo_present(frame, logos):
            start += 1
            break
    
    # Recherche avant pour la transition absent->présent
    end = end_kf
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while end < total_frames - 1:
        end += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, end)
        ret, frame = cap.read()
        if ret and is_logo_present(frame, logos):
            end -= 1
            break
    
    return (start, end)

"""
def get_keyframe_timestamps(video_path: str) -> List[float]:
    ""
    Utilise ffprobe pour récupérer la liste des timestamps (en secondes)
    de toutes les keyframes (frames de type I) de la vidéo.
    ""
    # Commande ffprobe
    #   -select_streams v:0 : on sélectionne le flux vidéo principal
    #   -skip_frame nokey   : on ignore toutes les frames qui NE sont PAS des keyframes
    #   -show_frames        : on affiche les infos sur chaque frame
    #   -show_entries frame=pkt_pts_time : on ne veut que le champ pkt_pts_time (timestamp)
    #   -of csv=print_section=0 : on formate la sortie en CSV (sans entête supplémentaire)
    #   -i video_path       : chemin de la vidéo
    #
    # => La sortie sera une liste de timestamps en secondes (souvent décimales).
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
"""

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
    fps: float
) -> Tuple[float, float]:
    """
    Étant donné qu'à l'index `idx_missing_logo` dans la liste `keyframes` le logo est absent,
    on cherche :
      - La keyframe juste AVANT qui serait la dernière avec logo présent
      - La keyframe juste APRÈS qui serait la première avec logo présent (fin de la pub)
    
    On renvoie un tuple (start_time_sec, end_time_sec).
    """
    total_duration = cap.get(cv2.CAP_PROP_DURATION)

    # 1) Trouver la dernière keyframe AVANT idx_missing_logo où le logo était présent
    start_key_idx = idx_missing_logo
    while start_key_idx >= 0:
        t = keyframes[start_key_idx]
        # On lit la vidéo à ce timestamp (en secondes)
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
    # start_key_idx : la dernière keyframe avec logo → la pub commence juste après
    # end_key_idx   : la première keyframe avec logo → la pub se termine juste avant
    if start_key_idx < 0:
        # Logo absent dès le début de la vidéo
        pub_start_sec = 0.0
    else:
        # On prend la keyframe correspondante et on considère que la pub commence
        # juste après cette keyframe => petite marge (p.ex. 1 frame)
        pub_start_sec = keyframes[start_key_idx]

    if end_key_idx >= len(keyframes):
        # Logo absent jusqu'à la fin de la vidéo
        pub_end_sec = total_duration
    else:
        pub_end_sec = keyframes[end_key_idx]

    return (pub_start_sec, pub_end_sec)

def detect_ads_by_keyframes(
    cap: cv2.VideoCapture,
    logos: List[np.ndarray],
    keyframes: List[float],
    fps: float
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
            pub_start_sec, pub_end_sec = find_ad_segment_boundaries(cap, logos, keyframes, idx, fps)
            pub_start_frame = int(pub_start_sec * fps)
            pub_end_frame   = int(pub_end_sec   * fps)

            if pub_end_frame > pub_start_frame and (pub_end_frame - pub_start_frame) >= min_ad_duration_frames:
                ad_segments.append((pub_start_sec, pub_end_sec))
                # On avance idx pour ne pas retomber sur la même séquence
                # On cherche la keyframe correspondant à pub_end_sec
                # comme on ne sait pas si c’est strictement un timestamp de la liste,
                # on saute simplement jusqu’à la première keyframe > pub_end_sec
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
            # On considère un chevauchement s’il y a recouvrement ou contiguïté
            if seg[0] <= last_seg[1]:
                merged_segments[-1] = (last_seg[0], max(last_seg[1], seg[1]))
            else:
                merged_segments.append(seg)

    return merged_segments

# -------------------------------------------------------------
# FONCTION PRINCIPALE
# -------------------------------------------------------------
def old_main(video_path: str, logo_folder: str):
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
    #duration_sec = cap.get(cv2.CAP_PROP_DURATION)
    duration_sec = total_frames / fps

    # Récupération de la liste des timestamps (en secondes) des keyframes
    keyframes = get_keyframe_timestamps(video_path)
    if not keyframes:
        print("[AVERTISSEMENT] Aucune keyframe extraite ou échec de ffprobe. " 
              "Impossible de poursuivre la détection.")
        return

    print(f"Vidéo : {video_path}")
    print(f"- FPS = {fps}")
    print(f"- Nombre total de frames = {total_frames}")
    print(f"- Durée (secondes) = {duration_sec:.1f}")
    print(f"- Nombre de keyframes = {len(keyframes)}")

    # 3) Détection des pubs via keyframes
    ad_segments = detect_ads_by_keyframes(cap, logos, keyframes, fps)

    cap.release()

    # 4) Affichage des segments publicitaires trouvés
    if not ad_segments:
        print("Aucune publicité détectée après analyse (keyframes).")
        return

    print("\nSegments publicitaires détectés (basés sur keyframes) :")
    for (start_sec, end_sec) in ad_segments:
        start_frame = int(start_sec * fps)
        end_frame   = int(end_sec   * fps)
        start_tc = frames_to_timecode(start_frame, fps)
        end_tc   = frames_to_timecode(end_frame,   fps)
        duration = end_sec - start_sec
        print(f" - Pub de {start_tc} à {end_tc} (Durée: {duration:.1f} s)")

def main(video_path: str, logo_folder: str):
    # Chargement des logos
    logos = load_logo_images(logo_folder)
    if not logos:
        print("[ERREUR] Aucun logo n'a été chargé.")
        return

    # Récupération des keyframes
    keyframes = get_keyframes(video_path)
    if not keyframes:
        print("Aucune keyframe détectée dans la vidéo.")
        return

    # Ouverture de la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Analyse des keyframes
    ad_segments = process_keyframes(cap, logos, keyframes, fps)
    
    # Affinage des segments détectés
    refined_segments = []
    for start_kf, end_kf in ad_segments:
        start, end = refine_segment(cap, logos, start_kf, end_kf)
        if end > start:
            refined_segments.append((start, end))
    
    cap.release()

    # Fusion des segments et filtrage
    merged = []
    for seg in sorted(refined_segments):
        if not merged:
            merged.append(seg)
        else:
            last = merged[-1]
            if seg[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], seg[1]))
            else:
                merged.append(seg)

    # Affichage des résultats
    print("\nSegments publicitaires détectés :")
    for start, end in merged:
        start_tc = frames_to_timecode(start, fps)
        end_tc = frames_to_timecode(end, fps)
        duration = (end - start) / fps
        print(f" - {start_tc} --> {end_tc} (Durée: {duration:.1f}s)")

# -------------------------------------------------------------
# POINT D'ENTRÉE
# -------------------------------------------------------------
if __name__ == "__main__":
    """
    Utilisation :
      python detect-ads-by-keyframes.py <chemin_vers_video> <dossier_logos>
    Exemple :
      python detect-ads-by-keyframes.py video.mp4 logos/
    """
    import sys
    if len(sys.argv) < 3:
        print("Usage: python detect-ads-by-keyframes.py <video_path> <logo_folder>")
        sys.exit(1)

    video_path_arg = sys.argv[1]
    logo_folder_arg = sys.argv[2]
    main(video_path_arg, logo_folder_arg)


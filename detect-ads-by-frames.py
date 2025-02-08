#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from typing import List, Tuple

# -------------------------------------------------------------
# CONSTANTES ET PARAMÈTRES
# -------------------------------------------------------------
LOGO_COORDS = (1740, 65, 85, 95)   # (x, y, largeur, hauteur)
LOGO_THRESHOLD = 0.8              # Seuil pour matchTemplate

SAMPLE_STEP = 15000               # Analyse une frame toutes les 15 000 frames
COARSE_STEP = 8000                # Pas pour la recherche "coarse" (avant affinage)
HALVING_STEPS = [4000, 2000, 1000, 500, 200, 100, 50, 25, 10, 5, 2, 1]

MIN_AD_DURATION_SECONDS = 180     # Durée minimale d'une pub en secondes (ici 3 minutes)

# -------------------------------------------------------------
# FONCTIONS DE CHARGEMENT ET DÉTECTION DU LOGO
# -------------------------------------------------------------
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
    # Vérification que la zone demandée se trouve dans la frame
    if x + w > frame_w or y + h > frame_h:
        return False

    roi = frame[y:y+h, x:x+w]
    # Conversion en niveaux de gris pour accélérer la comparaison
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    for logo in logos:
        # Si besoin, redimensionner le logo pour qu'il corresponde exactement à la ROI
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
    Convertit le numéro de frame en timecode au format hh:mm:ss.
    """
    total_seconds = frame_index / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# -------------------------------------------------------------
# FONCTIONS DE RECHERCHE "COARSE" ET D'AFFINAGE
# -------------------------------------------------------------
def find_previous_logo_frame(cap: cv2.VideoCapture, logos: List[np.ndarray],
                             ref_frame: int, step: int) -> int:
    """
    À partir de 'ref_frame' (où le logo est absent), recule par pas de 'step'
    pour trouver la dernière frame où le logo était présent.
    """
    current = ref_frame
    while current >= 0:
        candidate = current - step
        if candidate < 0:
            return 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ret, frame = cap.read()
        if not ret:
            return 0
        if is_logo_present(frame, logos):
            return candidate
        current = candidate
    return 0

def find_next_logo_frame(cap: cv2.VideoCapture, logos: List[np.ndarray],
                         ref_frame: int, step: int, total_frames: int) -> int:
    """
    À partir de 'ref_frame' (où le logo est absent), avance par pas de 'step'
    pour trouver la première frame où le logo réapparaît.
    """
    current = ref_frame
    while current < total_frames:
        candidate = current + step
        if candidate >= total_frames:
            return total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ret, frame = cap.read()
        if not ret:
            return total_frames - 1
        if is_logo_present(frame, logos):
            return candidate
        current = candidate
    return total_frames - 1

def refine_boundary_present_to_absent(cap: cv2.VideoCapture, logos: List[np.ndarray],
                                      low_frame: int, high_frame: int,
                                      steps: List[int]) -> int:
    """
    Affine la limite de transition de logo PRÉSENT à ABSENT.
    On suppose qu'à 'low_frame' le logo est présent et à 'high_frame' il est absent.
    On descend progressivement les pas de recherche pour trouver la première frame
    où le logo disparaît.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    low = max(low_frame, 0)
    high = min(high_frame, total_frames - 1)

    for step in steps:
        while (high - low) > step:
            mid = (low + high) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            if not ret:
                break
            if is_logo_present(frame, logos):
                low = mid  # le logo est encore présent → la transition est après
            else:
                high = mid  # le logo est absent → la transition se trouve avant ou à mid
    return high

def refine_boundary_absent_to_present(cap: cv2.VideoCapture, logos: List[np.ndarray],
                                      low_frame: int, high_frame: int,
                                      steps: List[int]) -> int:
    """
    Affine la limite de transition de logo ABSENT à PRÉSENT.
    On suppose qu'à 'low_frame' le logo est absent et à 'high_frame' il est présent.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    low = max(low_frame, 0)
    high = min(high_frame, total_frames - 1)

    for step in steps:
        while (high - low) > step:
            mid = (low + high) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            if not ret:
                break
            if is_logo_present(frame, logos):
                high = mid  # le logo est présent → la transition se trouve avant mid
            else:
                low = mid   # le logo est absent → la transition est après mid
    return high

def find_ad_segment(cap: cv2.VideoCapture, logos: List[np.ndarray],
                    ref_frame: int, total_frames: int) -> Tuple[int, int]:
    """
    À partir d'une frame de référence 'ref_frame' où le logo est absent (début d'une pub),
    on procède en 3 étapes :
      1. Recherche "coarse" vers l'arrière pour trouver la dernière frame où le logo était présent.
      2. Recherche "coarse" vers l'avant pour trouver la première frame où le logo réapparaît.
      3. Affinage de ces limites (transition présent->absent et absent->présent).
    Renvoie un tuple (pub_start, pub_end).
    """
    # 1) Reculer pour trouver la dernière frame où le logo est présent
    start_coarse = find_previous_logo_frame(cap, logos, ref_frame, COARSE_STEP)
    # 2) Avancer pour trouver la première frame où le logo réapparaît
    end_coarse = find_next_logo_frame(cap, logos, ref_frame, COARSE_STEP, total_frames)
    # 3) Affiner précisément la transition :
    pub_start = refine_boundary_present_to_absent(cap, logos, start_coarse, ref_frame, HALVING_STEPS)
    pub_end   = refine_boundary_absent_to_present(cap, logos, ref_frame, end_coarse, HALVING_STEPS)
    return (pub_start, pub_end)

# -------------------------------------------------------------
# ANALYSE DE LA VIDÉO POUR DÉTECTER LES PUBS
# -------------------------------------------------------------
def process_pass(cap: cv2.VideoCapture, logos: List[np.ndarray],
                 total_frames: int, start_frame: int, step: int,
                 ad_segments: List[Tuple[int, int]], fps: float,
                 min_ad_duration_frames: int) -> None:
    """
    Parcourt la vidéo en partant de 'start_frame' et en avançant par incréments de 'step'.
    Dès qu'une frame où le logo est absent est détectée, on cherche à délimiter précisément
    le segment publicitaire (début et fin). Le segment n'est retenu que si sa durée dépasse
    la durée minimale (ici 3 minutes).
    """
    current_frame = start_frame
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # Si dans la zone le logo n'est pas détecté → potentielle pub
        if not is_logo_present(frame, logos):
            pub_start, pub_end = find_ad_segment(cap, logos, current_frame, total_frames)
            if pub_end > pub_start and (pub_end - pub_start) >= min_ad_duration_frames:
                ad_segments.append((pub_start, pub_end))
                # Pour éviter des détections multiples sur le même segment,
                # on saute au-delà de la pub détectée en ajoutant 'step'
                current_frame = pub_end + step
            else:
                current_frame += step
        else:
            current_frame += step

# -------------------------------------------------------------
# FONCTION PRINCIPALE
# -------------------------------------------------------------
def main(video_path: str, logo_folder: str):
    """
    1. Charge les images de référence du logo.
    2. Ouvre la vidéo et récupère ses paramètres (nombre de frames, FPS, etc.).
    3. Analyse la vidéo en deux passes (démarrage à 0 puis à 7500 si aucune pub n'est détectée)
       pour repérer les segments publicitaires en se basant sur la détection du logo.
    4. Affiche les segments (avec timecode et durée).
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
    min_ad_duration_frames = int(fps * MIN_AD_DURATION_SECONDS)

    print(f"Vidéo : {video_path}")
    print(f"- FPS = {fps}")
    print(f"- Nombre total de frames = {total_frames}")
    print(f"- Durée minimale d'une pub (en frames) = {min_ad_duration_frames}")

    ad_segments = []  # Liste des segments publicitaires détectés sous forme de tuples (start_frame, end_frame)

    # 3) Première passe : à partir de la frame 0, avec un pas de 15 000 frames
    process_pass(cap, logos, total_frames, 0, SAMPLE_STEP, ad_segments, fps, min_ad_duration_frames)

    # Si aucune pub n'a été détectée lors de la première passe,
    # on relance l'analyse en démarrant à la 7500ème frame (soit SAMPLE_STEP // 2)
    if not ad_segments:
        print("Aucune pub détectée lors de la première passe. Activation de la seconde passe...")
        process_pass(cap, logos, total_frames, SAMPLE_STEP // 2, SAMPLE_STEP, ad_segments, fps, min_ad_duration_frames)

    cap.release()

    # 4) Fusion optionnelle des segments qui pourraient se chevaucher
    ad_segments.sort()
    merged_segments = []
    for seg in ad_segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            last_seg = merged_segments[-1]
            if seg[0] <= last_seg[1]:  # chevauchement
                merged_segments[-1] = (last_seg[0], max(last_seg[1], seg[1]))
            else:
                merged_segments.append(seg)

    # Affichage des segments publicitaires trouvés
    if not merged_segments:
        print("Aucune publicité détectée après analyse.")
        return

    print("\nSegments publicitaires détectés :")
    for start_frame, end_frame in merged_segments:
        start_tc = frames_to_timecode(start_frame, fps)
        end_tc = frames_to_timecode(end_frame, fps)
        duration = (end_frame - start_frame) / fps
        print(f" - Pub de {start_tc} à {end_tc} (Durée: {duration:.1f} s)")

# -------------------------------------------------------------
# POINT D'ENTRÉE
# -------------------------------------------------------------
if __name__ == "__main__":
    """
    Utilisation :
      python detect-ads-by-frames.py <chemin_vers_video> <dossier_logos>
    Exemple :
      python detect-ads-by-frames.py video.mp4 logos/
    """
    import sys
    if len(sys.argv) < 3:
        print("Usage: python detect-ads-by-frames.py <video_path> <logo_folder>")
        sys.exit(1)

    video_path_arg = sys.argv[1]
    logo_folder_arg = sys.argv[2]
    main(video_path_arg, logo_folder_arg)



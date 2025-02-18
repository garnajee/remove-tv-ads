#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script permettant de détecter une pub.
La position du logo est détectetée approxiamtivement.
Les résultats sont corrects mais non-précis.
"""

import cv2
import os
import numpy as np
from typing import List, Tuple

# -------------------------------------------------------------
# CONSTANTES ET PARAMÈTRES GLOBAUX
# -------------------------------------------------------------
SAMPLE_STEP = 15000       # Échantillonnage principal (toutes les 15 000 frames)
COARSE_STEP = 8000        # Recherche "coarse" initiale (on recule/avance par 8000 frames)
HALVING_STEPS = [4000, 2000, 1000, 500, 200, 100, 50, 25, 10, 5, 2, 1]
LOGO_THRESHOLD = 0.8      # Seuil de détection pour matchTemplate (à ajuster)
TOP_RIGHT_RATIO = 0.4     # Fraction de la largeur pour la zone en haut à droite
TOP_HEIGHT_RATIO = 0.4    # Fraction de la hauteur pour la zone en haut


# -------------------------------------------------------------
# FONCTIONS UTILITAIRES
# -------------------------------------------------------------
def load_logo_images(logo_folder: str) -> List[np.ndarray]:
    """
    Charge toutes les images (png, jpg...) d'un dossier donné comme logos de référence.
    Retourne une liste de tableaux numpy (images BGR).
    """
    logos = []
    for file_name in os.listdir(logo_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(logo_folder, file_name)
            logo_img = cv2.imread(path)
            if logo_img is not None:
                logos.append(logo_img)
    return logos


def is_logo_present(frame: np.ndarray, logos: List[np.ndarray], threshold: float = LOGO_THRESHOLD) -> bool:
    """
    Détermine si l'un des logos de référence est présent dans la frame (True/False).
    On utilise matchTemplate (TM_CCOEFF_NORMED) dans la zone en haut à droite (ROI).
    """
    if frame is None:
        return False

    h, w = frame.shape[:2]

    # ROI : zone en haut à droite
    x1 = int(w * (1 - TOP_RIGHT_RATIO))
    y1 = 0
    x2 = w
    y2 = int(h * TOP_HEIGHT_RATIO)
    roi = frame[y1:y2, x1:x2]

    roi_h, roi_w = roi.shape[:2]

    for logo in logos:
        logo_h, logo_w = logo.shape[:2]
        
        # Si le logo est plus grand que la ROI, on peut l'ignorer ou le redimensionner
        if logo_h > roi_h or logo_w > roi_w:
            continue

        res = cv2.matchTemplate(roi, logo, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
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
# FONCTIONS DE RECHERCHE "COARSE"
# -------------------------------------------------------------
def find_previous_logo_frame(cap: cv2.VideoCapture,
                             logos: List[np.ndarray],
                             ref_frame: int,
                             step: int) -> int:
    """
    À partir de ref_frame (où le logo est absent),
    on recule par pas de `step` pour trouver la première frame où le logo est présent.
    Renvoie l'index de cette frame (ou 0 si on n'en trouve pas).
    """
    current = ref_frame
    while current >= 0:
        next_candidate = current - step
        if next_candidate < 0:
            # On atteint le début de la vidéo
            return 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, next_candidate)
        ret, frame = cap.read()
        if not ret:
            # Problème de lecture -> on s'arrête et renvoie 0
            return 0
        
        if is_logo_present(frame, logos):
            # On a trouvé une frame où le logo est présent
            return next_candidate
        else:
            # Continue à reculer
            current = next_candidate

    return 0


def find_next_logo_frame(cap: cv2.VideoCapture,
                         logos: List[np.ndarray],
                         ref_frame: int,
                         step: int,
                         total_frames: int) -> int:
    """
    À partir de ref_frame (logo absent),
    on avance par pas de `step` pour trouver la première frame où le logo est présent.
    Renvoie l'index de cette frame (ou total_frames-1 si on n'en trouve pas).
    """
    current = ref_frame
    while current < total_frames:
        next_candidate = current + step
        if next_candidate >= total_frames:
            return total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, next_candidate)
        ret, frame = cap.read()
        if not ret:
            return total_frames - 1
        
        if is_logo_present(frame, logos):
            return next_candidate
        else:
            current = next_candidate

    return total_frames - 1


# -------------------------------------------------------------
# AFFINAGE PAR DIVISION DE PAS ("HALVING")
# -------------------------------------------------------------
def refine_boundary_present_to_absent(cap: cv2.VideoCapture,
                                      logos: List[np.ndarray],
                                      low_frame: int,
                                      high_frame: int,
                                      steps: List[int]) -> int:
    """
    Cherche la PREMIÈRE frame dans l'intervalle [low_frame, high_frame]
    où le logo devient ABSENT (transition de PRESENT à ABSENT).

    Hypothèses :
      - À low_frame, le logo est PRÉSENT
      - À high_frame, le logo est ABSENT
    On descend les pas dans 'steps' (ex: 4000,2000,1000, etc.) pour trouver la transition.
    On renvoie l'index de la frame où l'absence se produit pour la 1ère fois.
    """
    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if low_frame < 0: low_frame = 0
    if high_frame >= cap_frames: high_frame = cap_frames - 1

    current_low = low_frame
    current_high = high_frame

    for step in steps:
        while (current_high - current_low) > step:
            mid = (current_low + current_high) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            if not ret:
                break

            if is_logo_present(frame, logos):
                # Le logo est encore présent -> la transition se trouve APRES mid
                current_low = mid
            else:
                # Le logo est absent -> la transition est AVANT ou à mid
                current_high = mid

    # À la fin, current_high est proche de la transition.
    return current_high


def refine_boundary_absent_to_present(cap: cv2.VideoCapture,
                                      logos: List[np.ndarray],
                                      low_frame: int,
                                      high_frame: int,
                                      steps: List[int]) -> int:
    """
    Cherche la PREMIÈRE frame dans [low_frame, high_frame]
    où le logo REdevient PRÉSENT (transition de ABSENT à PRÉSENT).

    Hypothèses :
      - À low_frame, le logo est ABSENT
      - À high_frame, le logo est PRÉSENT
    On descend les pas (ex: 4000, 2000, 1000...) pour trouver cette transition.
    """
    cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if low_frame < 0: low_frame = 0
    if high_frame >= cap_frames: high_frame = cap_frames - 1

    current_low = low_frame
    current_high = high_frame

    for step in steps:
        while (current_high - current_low) > step:
            mid = (current_low + current_high) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            if not ret:
                break

            if is_logo_present(frame, logos):
                # Logo présent -> transition se trouve AVANT mid
                current_high = mid
            else:
                # Logo absent -> transition est APRES mid
                current_low = mid

    return current_high


# -------------------------------------------------------------
# DÉTECTION D'UN SEGMENT DE PUB
# -------------------------------------------------------------
def find_ad_segment(cap: cv2.VideoCapture,
                    logos: List[np.ndarray],
                    ref_frame: int,
                    total_frames: int) -> Tuple[int, int]:
    """
    On sait qu'à 'ref_frame', le logo est ABSENT (pub).
    1) On recule de COARSE_STEP pour trouver la frame "logo présent" la plus proche avant la pub.
       => start_coarse
    2) On avance de COARSE_STEP pour trouver la frame "logo présent" la plus proche après la pub.
       => end_coarse
    3) On utilise un affinage (HALVING_STEPS) pour trouver :
       - pub_start = transition (présent -> absent)
       - pub_end   = transition (absent -> présent)
    """

    # 1) TROUVER LE "start_coarse" = la frame la plus proche (en reculant) où le logo est présent
    start_coarse = find_previous_logo_frame(cap, logos, ref_frame, COARSE_STEP)
    
    # 2) TROUVER LE "end_coarse" = la frame la plus proche (en avançant) où le logo est présent
    end_coarse = find_next_logo_frame(cap, logos, ref_frame, COARSE_STEP, total_frames)

    # Vérifier qu'on lit bien (start_coarse < end_coarse)
    # Hypothèse: start_coarse => logo présent, ref_frame => logo absent, end_coarse => logo présent

    # A) AFFINER LE DÉBUT DE LA PUB = transition PRÉSENT -> ABSENT
    #    => [start_coarse, ref_frame]
    #    on suppose start_coarse a le logo présent, ref_frame est absent
    pub_start = refine_boundary_present_to_absent(cap, logos, start_coarse, ref_frame, HALVING_STEPS)

    # B) AFFINER LA FIN DE LA PUB = transition ABSENT -> PRÉSENT
    #    => [ref_frame, end_coarse]
    pub_end   = refine_boundary_absent_to_present(cap, logos, ref_frame, end_coarse, HALVING_STEPS)

    return (pub_start, pub_end)


# -------------------------------------------------------------
# FONCTION PRINCIPALE
# -------------------------------------------------------------
def main(video_path: str, logo_folder: str):
    """
    - Ouvre la vidéo
    - Charge les logos
    - Recherche des segments publicitaires
    - Affiche la liste des segments (début et fin)
    """
    # 1) CHARGER LES LOGOS
    logos = load_logo_images(logo_folder)
    if not logos:
        print("[ERREUR] Aucune image de logo n'a été chargée. Vérifiez le dossier.")
        return
    
    # 2) OUVRIR LA VIDÉO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo : {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Vidéo : {video_path}")
    print(f"- FPS = {fps}")
    print(f"- Nombre total de frames = {total_frames}")
    
    ad_segments = []      # liste de tuples (start_frame, end_frame)
    current_frame = 0
    
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1) Test si le logo est absent
        if not is_logo_present(frame, logos):
            # => On est en pub
            pub_start, pub_end = find_ad_segment(cap, logos, current_frame, total_frames)
            
            # On évite de stocker un segment incohérent
            if pub_end >= pub_start:
                ad_segments.append((pub_start, pub_end))

                # On saute au-delà de la pub
                current_frame = pub_end + SAMPLE_STEP
            else:
                current_frame += SAMPLE_STEP
        else:
            # Logo présent => on saute SAMPLE_STEP
            current_frame += SAMPLE_STEP
    
    cap.release()

    # --------------------------------------------------------
    # AFFICHAGE DES SEGMENTS TROUVÉS
    # --------------------------------------------------------
    if not ad_segments:
        print("Aucune publicité détectée.")
        return
    
    print("\nSegments publicitaires détectés :")
    for (start_f, end_f) in ad_segments:
        start_tc = frames_to_timecode(start_f, fps)
        end_tc = frames_to_timecode(end_f, fps)
        print(f" - Pub entre {start_tc} et {end_tc}")


# -------------------------------------------------------------
# Point d'entrée standard
# -------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage:
      python detect_ads.py <video_path> <logo_folder>
    """
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python detect_ads.py <video_path> <logo_folder>")
        sys.exit(1)
    
    video_path_arg = sys.argv[1]
    logo_folder_arg = sys.argv[2]
    
    main(video_path_arg, logo_folder_arg)


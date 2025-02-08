# remove-tv-ads

Supprimer les pub d'une émission.

Objectif : que ça reste rapide/performant. Faut penser que les émissions (pub comprises évidemment) peuvent durer 4h ou +.

lien de la vidéo d'exmple [ici](https://gofile.io/d/dYhgJ9)

## Tests

- Pour le moment que pour M6.

### Extrait de 49'

- coordonnées du logo déterminées à la main.

```bash
$ time python detect-ads-by-frame.py cpva2.mkv logos-true-pos
Vidéo : ../remove-tv-ads-local/output-mkv/cpva2.mkv
- FPS = 25.0
- Nombre total de frames = 73488
- Durée minimale d'une pub (en frames) = 4500
Aucune pub détectée lors de la première passe. Activation de la seconde passe...

Segments publicitaires détectés :
 - Pub de 00:11:20 à 00:18:33 (Durée: 432.3 s)
python detect-ads-by-frame.py cpva2.mkv   35,00s user 1,94s system 127% cpu 29,058 total
```

### Extrait de 19'

- la pub démarre à 2'19" (et quelques millisecondes en +) et fini à 10'25~26"

```bash
$ time python detect-ads3.py cpva.mkv logos-1
Vidéo : cpva.mkv
- FPS = 25.0
- Nombre total de frames = 28753

Segments publicitaires détectés :
 - Pub entre 00:02:19 et 00:10:26
python detect-ads3.py cpva.mkv logos  9,63s user 0,48s system 176% cpu 5,720 total
```

## TODO

- [ ] scanner sur les *keyframe* au lieu des frames
- [ ] scanner toutes les 15 000 (key)frame :
    - si pas de pub détecté, alors scanner toutes les 7500 + 15 000 (key)frame
- [ ] si pub inférieur à 1min, alors ignorer
- [ ] mieux détecter l'endroit où est le logo dans l'image. plus précis.
- [ ] ajouter secion tests plus précise

## Idées autres

- [ ] Trouver la pub en se basant sur le son du jingle.

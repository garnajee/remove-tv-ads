# remove-tv-ads

lien de la vidéo d'exmple [ici](https://gofile.io/d/dYhgJ9)

## Tests

- Pour le moment que pour M6.
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


# Architecture V2

La V2 garde le POC YouTube exécutable, mais ajoute une couche de production légère autour de lui. L'objectif est de passer d'un notebook démonstrateur à un observatoire reproductible, auditables et compatible avec une extension Matomo.

## Modules

- `observatoire.config` charge les corpus vidéo et les lexiques depuis JSON/YAML.
- `observatoire.cache` ajoute un manifeste au cache brut pour éviter de réutiliser des données collectées avec des paramètres différents.
- `observatoire.privacy` bloque les exports contenant des colonnes d'identification auteur directes.
- `observatoire.cli` fournit une entrée V2 avec `python -m observatoire.cli`.
- `observatoire_youtube_poc.py` reste compatible avec les commandes historiques.

## Contrat De Données

Le cache brut repose sur trois fichiers dans `data/` :

- `youtube_comments_raw_anonymized.csv`
- `video_metadata.csv`
- `raw_collection_manifest.json`

Le manifeste encode les paramètres qui changent le périmètre de collecte :

- liste des `video_id`
- `max_comments_per_video`
- `include_replies`
- collecteur utilisé

Si l'un de ces paramètres change, la collecte est relancée au lieu de réutiliser silencieusement un cache incompatible.

## Qualité Sémantique

Les commentaires restent tous présents dans `data/youtube_comments_enriched.csv`, mais seuls les commentaires suffisamment discursifs participent aux embeddings et aux clusters. Le pipeline exclut par défaut :

- les commentaires sans tokens alphabétiques, par exemple emoji-only ;
- les réactions courtes centrées sur des noms ou encouragements ;
- les commentaires trop courts pour produire un cadrage interprétable.

Les seuils peuvent être ajustés avec `--min-cluster-chars` et `--min-cluster-meaningful-tokens`. Le fichier `outputs/semantic_filter_summary.csv` résume les exclusions par raison.

## Prochaines Étapes

- Déplacer progressivement les fonctions de `observatoire_youtube_poc.py` vers `collectors/`, `processing/`, `frames/`, `models/` et `reports/`.
- Ajouter un import Matomo agrégé pour relier contenus, UTM, CTA et conversions sans scoring individuel.
- Ajouter un flux d'annotation humaine assistée pour valider les cadrages et entraîner un classifieur supervisé.
- Ajouter un rapport qualité de données par run : vidéos sans commentaires, erreurs API, langues, doublons, volume par acteur.

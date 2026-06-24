# Observatoire des cadrages politiques à partir de commentaires YouTube

POC exécutable dans Google Colab ou en local pour collecter des commentaires publics via l’API officielle **YouTube Data API v3**, les anonymiser, puis produire une analyse agrégée des cadrages politiques observables dans les conversations.

Ce projet produit uniquement des agrégats : thèmes, cadrages, clusters sémantiques, trajectoires et distances de réception.

## Structure V2

La V2 conserve le script POC historique et ajoute un petit package `observatoire/` pour rendre le pipeline plus robuste :

```text
.
├── observatoire_youtube_poc.py
├── observatoire/
│   ├── cache.py
│   ├── cli.py
│   ├── config.py
│   ├── privacy.py
│   └── schemas.py
├── config/
│   ├── corpus.example.json
│   └── frames.example.json
├── docs/
│   └── V2_ARCHITECTURE.md
└── tests/
```

Le script `observatoire_youtube_poc.py` reste exécutable comme avant. L'entrée V2 recommandée est :

```bash
python -m observatoire.cli --config config/corpus.example.json --max-comments-per-video 100
```

## Structure générée

```text
.
├── observatoire_youtube_poc.py
├── observatoire_youtube_poc_colab.ipynb
├── requirements.txt
├── README.md
├── data/
│   ├── youtube_comments_raw_anonymized.csv
│   ├── youtube_comments_enriched.csv
│   ├── video_metadata.csv
│   └── raw_collection_manifest.json
└── outputs/
    ├── frame_actor_matrix.csv
    ├── frame_time_series.csv
    ├── semantic_clusters.csv
    ├── semantic_filter_summary.csv
    ├── reception_distance_by_video.csv
    ├── semantic_actor_trajectories.csv
    ├── interpretation_note.md
    └── observatoire_cadrages_youtube_dashboard.html
```

Les dossiers `data/` et `outputs/` sont créés automatiquement au lancement.

## Créer une clé YouTube Data API

1. Aller dans [Google Cloud Console](https://console.cloud.google.com/).
2. Créer ou sélectionner un projet.
3. Activer **YouTube Data API v3** dans `APIs & Services`.
4. Créer une clé dans `Credentials`.
5. Restreindre la clé si possible : API YouTube Data API v3 uniquement, restrictions adaptées à votre environnement.

## Renseigner `YOUTUBE_API_KEY`

### Dans Google Colab

Option recommandée :

1. Ouvrir le notebook `observatoire_youtube_poc_colab.ipynb`.
2. Dans le panneau **Secrets**, ajouter une variable nommée `YOUTUBE_API_KEY`.
3. Autoriser le notebook à y accéder.

Option rapide :

```python
import os
os.environ["YOUTUBE_API_KEY"] = "VOTRE_CLE_API"
```

Éviter de publier un notebook contenant une clé en clair.

### En local

```bash
export YOUTUBE_API_KEY="VOTRE_CLE_API"
python observatoire_youtube_poc.py --max-comments-per-video 500
```

## Modifier la liste des vidéos

Option V2 recommandée : copier `config/corpus.example.json`, puis lancer le pipeline avec `--config` ou `--videos-config`.

```bash
python -m observatoire.cli --config config/corpus.example.json
python observatoire_youtube_poc.py --videos-config config/corpus.example.json
```

Option POC historique : modifier la variable `VIDEOS` dans `observatoire_youtube_poc.py` ou dans le notebook.

Chaque entrée accepte :

```python
{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "actor": "LFI",
    "sequence": "pouvoir_achat",
    "label": "Titre humain optionnel"
}
```

Le code accepte aussi :

- URL YouTube classique ;
- URL courte `youtu.be` ;
- URL `embed`, `shorts` ou `live` ;
- identifiant vidéo brut de 11 caractères.

## Lancer la collecte

En local :

```bash
pip install -r requirements.txt
export YOUTUBE_API_KEY="VOTRE_CLE_API"
python observatoire_youtube_poc.py --max-comments-per-video 500
```

Avec les réponses aux commentaires :

```bash
python observatoire_youtube_poc.py --max-comments-per-video 500 --include-replies
```

Ignorer le cache et rappeler l’API :

```bash
python observatoire_youtube_poc.py --force-refresh
```

Utiliser une liste externe JSON/YAML :

```bash
python observatoire_youtube_poc.py --videos-config videos.json
```

## Cache et quotas

Le POC écrit un cache local :

- `data/youtube_comments_raw_anonymized.csv`
- `data/video_metadata.csv`
- `data/raw_collection_manifest.json`

Le manifeste encode la liste des vidéos, `max_comments_per_video`, `include_replies` et le collecteur. Si ces paramètres changent, le script relance la collecte au lieu de réutiliser silencieusement un cache incompatible.

La collecte est limitée par défaut à `500` commentaires par vidéo. Vous pouvez réduire cette limite pendant les tests :

```bash
python observatoire_youtube_poc.py --max-comments-per-video 100
```

Erreurs gérées explicitement :

- clé API absente ;
- clé invalide ;
- quota dépassé ;
- commentaires désactivés ;
- vidéo introuvable ou inaccessible.

## Analyses incluses

- Nettoyage de texte en conservant les accents.
- Détection de langue légère et optionnelle avec `langdetect`.
- Cadrages lexicaux interprétables.
- Tonalité / stance lexicale prudente : adhésion, rejet, ironie, scepticisme, incompréhension, déplacement du débat.
- Matrices `acteur × cadrage`, `semaine × cadrage`, `vidéo × cadrage`.
- Embeddings multilingues avec `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Filtrage des commentaires trop courts, emoji-only ou réactionnels avant clustering sémantique.
- Réduction 2D par PCA, UMAP si disponible.
- Clustering KMeans robuste pour petit volume.
- Mots caractéristiques par cluster via TF-IDF.
- Exemples anonymisés proches des centroïdes.
- Trajectoires sémantiques par acteur et semaine.
- Distance entre embedding du titre vidéo et embedding moyen des commentaires.
- Dashboard HTML autonome.
- Note d’interprétation Markdown prudente.

## Limites méthodologiques

Les commentaires YouTube ne sont pas représentatifs de l’opinion générale. Ils reflètent :

- une population auto-sélectionnée ;
- la visibilité algorithmique ;
- la modération des chaînes et de YouTube ;
- les effets de mobilisation ;
- les temporalités propres à chaque publication ;
- la tonalité très particulière des espaces de commentaires.

Les clusters sémantiques sont des régions statistiques émergentes, pas des vérités objectives. Les labels automatiques doivent être validés par lecture humaine.

Les cadrages lexicaux sont utiles comme baseline interprétable, mais ils ratent l’ironie, les sous-entendus, les formulations nouvelles et certains déplacements rhétoriques.

## Limites juridiques et RGPD

- Ne pas exporter les noms d’utilisateurs.
- Ne pas construire de scoring individuel.
- Ne pas réidentifier les auteurs.
- Limiter la durée de conservation des données.
- Respecter les conditions de l’API YouTube et les obligations RGPD applicables.
- Documenter la finalité : analyse agrégée de conversations publiques.

Par défaut, le POC ne conserve pas `author_display_name`. Si un identifiant auteur est présent, il est haché immédiatement dans `author_hash`.

## Interprétation prudente

La note générée dans `outputs/interpretation_note.md` emploie volontairement des formulations du type :

> Dans les commentaires collectés, on observe que…

Elle ne doit pas être reformulée en :

> Les gens pensent que…

Ce garde-fou est important : les données collectées ne mesurent ni l’électorat, ni l’opinion publique, ni les audiences complètes des chaînes.

## Suite V2

Socle posé :

- Configuration externe du corpus et des lexiques.
- Cache manifesté pour éviter les réutilisations incohérentes.
- Contrôle simple des colonnes sensibles avant export.
- Filtre de qualité sémantique avant embeddings/clusters.
- Tests unitaires sur config, cache et privacy.

Prochaines extensions :

- BERTopic pour des topics plus lisibles.
- HDBSCAN pour des clusters de densité sans fixer `k`.
- Détection de stance par modèle local ou API.
- Annotation humaine assistée.
- Comparaison avec données Matomo agrégées.
- Graphe dynamique des cadrages.
- Modèle markovien acteur-cadrage.
- Score agrégé de sortie de bulle.
- Détection de recodages adverses.

# Observatoire des cadrages politiques à partir de commentaires YouTube

POC exécutable dans Google Colab ou en local pour collecter des commentaires publics via l’API officielle **YouTube Data API v3**, les anonymiser, puis produire une analyse agrégée des cadrages politiques observables dans les conversations.

Ce projet ne sert pas au microciblage politique, au scoring individuel ni à la persuasion personnalisée. Il produit uniquement des agrégats : thèmes, cadrages, clusters sémantiques, trajectoires et distances de réception.

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

La V2.5 ajoute une extraction inductive optionnelle de claims par LLM :

```bash
export OPENAI_API_KEY="..."
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-claims \
  --cluster-claims \
  --label-claim-clusters
```

Elle peut aussi utiliser un modèle open source local via Ollama :

```bash
ollama pull llama3.1:8b
ollama serve

python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-claims \
  --cluster-claims \
  --label-claim-clusters \
  --llm-provider ollama \
  --llm-model llama3.1:8b
```

Si Ollama tourne ailleurs que sur `http://localhost:11434`, utilisez `--llm-base-url` ou `OLLAMA_BASE_URL`.

Pour un test rapide avec très peu de commentaires, les clusters peuvent rester vides : c'est volontairement plus prudent que de forcer des groupes bruités. Vous pouvez abaisser les seuils uniquement pour exploration :

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-claims \
  --cluster-claims \
  --llm-provider ollama \
  --llm-model llama3.1:8b \
  --max-comments-per-video 10 \
  --semantic-cluster-min-size 3 \
  --claim-cluster-min-size 3
```

La V2.6.1 ajoute l'option 1 de cartographie discursive : une fiche structurée par verbatim, produite par LLM sans fine-tuning, puis clusterisée.

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --cluster-discourse-cards \
  --llm-provider ollama \
  --llm-model llama3.1:8b \
  --discursive-card-limit 50
```

La V2.6.3 ajoute un graphe discursif typé construit à partir de ces fiches. Le graphe relie commentaires, frames, claims canoniques, cibles, stances, tonalités, source vidéo et période, puis produit des communautés discursives interprétables.

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --build-discourse-graph \
  --llm-provider ollama \
  --llm-model llama3.1:8b \
  --discursive-card-limit 50
```

Pour accélérer l'annotation locale avec Ollama, utilisez le prompt rapide, le cache LLM et éventuellement deux workers :

```bash
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_NUM_PREDICT=768

python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --build-discourse-graph \
  --llm-provider ollama \
  --llm-model llama3.1:8b \
  --discursive-card-prompt prompts/extract_discursive_card_fast.md \
  --discursive-card-workers 2
```

Les réponses LLM sont mises en cache dans `outputs/discursive_card_llm_cache.jsonl`. Une relance qui reconstruit seulement le graphe réutilise automatiquement `outputs/discursive_cards.csv` si le fichier existe :

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --build-discourse-graph \
  --llm-provider ollama \
  --llm-model llama3.1:8b
```

Pour forcer explicitement la réutilisation même si `--extract-discourse-cards` est présent :

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --build-discourse-graph \
  --reuse-discourse-cards
```

Voir `docs/LLM_ANNOTATION_SPEED.md` pour les stratégies de vitesse, les compromis qualité et les commandes de benchmark.

Pour un petit corpus, les communautés peuvent rester vides. Pour explorer seulement, abaissez `--discourse-graph-min-community-size` ou `--discourse-graph-similarity-threshold`.

### Post-traitement sans ré-encodage

Les scripts conservés dans `scripts/` couvrent uniquement les graphes, rapports et remappings utiles sans relancer le LLM. La documentation complète est dans `docs/GRAPH_POSTPROCESSING.md`.

Post-traiter le graphe discursif, générer les exports GEXF/HTML et recalculer une projection commentaire-commentaire filtrée :

```bash
python scripts/postprocess_discursive_graph.py \
  --input-dir outputs \
  --output-dir outputs/graph_postprocess \
  --include-similarity-edges \
  --similarity-threshold 0.85 \
  --drop-non-signal-labels
```

Visualiser les principaux ponts discursifs entre deux sources, par défaut RN et LFI :

```bash
python scripts/visualize_rn_lfi_bridges.py \
  --input-dir outputs \
  --output-dir outputs/rn_lfi_bridges \
  --source-a RN \
  --source-b LFI
```

Transformer ces ponts en rapport lisible et auditable :

```bash
python scripts/generate_bridge_report.py \
  --input-dir outputs \
  --bridge-dir outputs/rn_lfi_bridges \
  --output-dir outputs/rn_lfi_bridges
```

Ces scripts produisent des CSV d'audit, des diagnostics JSON, des exports GEXF et des HTML autonomes. Ils filtrent les labels vagues comme `other`/`unknown`, pénalisent les attributs trop fréquents et utilisent des tailles de nœuds dépendantes du degré ou du score de pont.

Scripts maintenus :

- `scripts/postprocess_discursive_graph.py` : graphe global, projection commentaire-commentaire, communautés post-traitées, diagnostics et HTML/GEXF.
- `scripts/visualize_rn_lfi_bridges.py` : sous-graphe des ponts entre deux sources, par défaut RN/LFI.
- `scripts/generate_bridge_report.py` : rapport lisible et auditable des ponts.
- `scripts/taxonomy_induction.py` : propositions de remapping taxonomique et rapport avant/après.

### Taxonomy induction and remapping

La taxonomie discursive doit rester stable, versionnée et comparable entre runs. Elle ne doit pas devenir opportuniste à chaque corpus. En revanche, les sorties existantes peuvent révéler des alias manquants, des sous-frames utiles ou des confusions d'axes (`frame` vs `tone` vs `rhetorical_register` vs `argument_family` vs `target`).

Le script `scripts/taxonomy_induction.py` ajoute une couche d'induction contrôlée sans relancer l'encodage LLM complet des commentaires. Il lit les fichiers déjà produits, propose des mappings auditables, crée une taxonomie candidate et applique uniquement les remappings suffisamment sûrs dans des colonnes séparées.

Commande heuristique, sans LLM :

```bash
python scripts/taxonomy_induction.py \
  --taxonomy config/discourse_taxonomy.example.json \
  --units outputs/discursive_units.csv \
  --nodes outputs/discursive_nodes.csv \
  --community-summary outputs/graph_postprocess/postprocessed_community_summary.csv \
  --output-dir outputs/taxonomy_induction \
  --mode heuristic
```

Commande avec suggestions LLM optionnelles :

```bash
python scripts/taxonomy_induction.py \
  --taxonomy config/discourse_taxonomy.example.json \
  --units outputs/discursive_units.csv \
  --nodes outputs/discursive_nodes.csv \
  --output-dir outputs/taxonomy_induction \
  --mode llm \
  --llm-provider openai \
  --llm-model gpt-4.1-mini
```

Sorties :

- `outputs/taxonomy_induction/taxonomy_induction_candidates.json` : propositions d'alias, sous-frames, déplacements d'axe et labels à garder en `other`.
- `outputs/taxonomy_induction/taxonomy_remap_table.csv` : table de correspondance avec justification, exemples et validation humaine requise ou non.
- `outputs/taxonomy_induction/discourse_taxonomy.candidate.json` : taxonomie candidate qui n'écrase jamais `config/discourse_taxonomy.example.json`.
- `outputs/taxonomy_induction/discursive_units_remapped.csv` : unités remappées avec colonnes `*_raw` et `*_remapped`.
- `outputs/taxonomy_induction/taxonomy_remap_report.md` : comparaison avant/après des taux de `other`, labels uniques et impact potentiel sur le graphe.

Après validation humaine de la table de remapping, relancez les scripts de post-processing graphe sur les sorties remappées ou intégrez les alias validés dans une future version contrôlée de la taxonomie. Le principe à conserver : taxonomie stable + induction contrôlée + validation humaine + remapping sans réencodage.

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
    ├── discursive_cards.csv
    ├── discursive_card_coverage.csv
    ├── discursive_card_llm_cache.jsonl
    ├── discursive_clusters.csv
    ├── discursive_units.csv
    ├── discursive_units.jsonl
    ├── discursive_nodes.csv
    ├── discursive_edges.csv
    ├── discursive_similarity_edges.csv
    ├── discursive_communities.csv
    ├── discursive_community_profiles.md
    ├── graph_postprocess/
    │   ├── postprocessed_comment_communities.csv
    │   ├── postprocessed_community_summary.csv
    │   ├── postprocessed_comment_projection_edges.csv
    │   ├── postprocessed_graph_diagnostics.json
    │   ├── discursive_full_graph.gexf
    │   ├── discursive_comment_projection.gexf
    │   ├── discursive_full_graph_by_source.html
    │   └── discursive_comment_projection_by_source.html
    ├── rn_lfi_bridges/
    │   ├── rn_lfi_attribute_bridges.csv
    │   ├── rn_lfi_direct_similarity_bridges.csv
    │   ├── rn_lfi_bridge_subgraph_nodes.csv
    │   ├── rn_lfi_bridge_subgraph_edges.csv
    │   ├── rn_lfi_bridge_diagnostics.json
    │   ├── rn_lfi_bridge_graph.gexf
    │   ├── rn_lfi_bridge_graph.html
    │   ├── bridge_report.csv
    │   └── bridge_report.md
    ├── taxonomy_induction/
    │   ├── taxonomy_induction_candidates.json
    │   ├── taxonomy_remap_table.csv
    │   ├── discourse_taxonomy.candidate.json
    │   ├── discursive_units_remapped.csv
    │   └── taxonomy_remap_report.md
    ├── discursive_incidence_frame.npz
    ├── discursive_incidence_claim.npz
    ├── discursive_incidence_stance.npz
    ├── comment_claims.csv
    ├── no_claim_summary.csv
    ├── claim_clusters.csv
    ├── claim_cluster_labels.md
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
- Garde-fou petit volume : les clusters sémantiques trop petits restent en bruit au lieu d'être forcés dans des familles artificielles.
- Fiches discursives structurées optionnelles par LLM : thème, cadrage, stance, argument, tonalité, ambiguïtés et citations.
- Taxonomie contrôlée des fiches discursives : `macro_frame`, `frame_primary`, stance normalisée, famille argumentative, tonalité, registre rhétorique et scores qualité.
- Clustering optionnel des fiches discursives pour cartographier des proximités de discours sans fine-tuning.
- Graphe discursif typé V2.6.3 : commentaires reliés à frames, claims canoniques, cibles, stances, tonalités, source vidéo et période.
- Communautés discursives interprétables à partir d'une similarité hybride par matrices d'incidence pondérées.
- Visualisation HTML interactive du graphe discursif complet, filtré et de la projection commentaire-commentaire.
- Post-traitement sans ré-encodage du graphe discursif, avec projection filtrée, diagnostics de hubs et exports GEXF/HTML.
- Visualisation des ponts RN/LFI ou entre deux sources configurables, à partir des attributs discursifs partagés et des similarités directes.
- Extraction inductive optionnelle de claims par LLM OpenAI ou Ollama, avec preuve textuelle obligatoire.
- Clustering optionnel des claims plutôt que des commentaires bruts.
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
- Extraction inductive de claims et clustering de claims.
- Fiches discursives LLM, graphe typé V2.6.3 et communautés discursives interprétables.
- Codebook contrôlé dans `config/discourse_taxonomy.example.json` et normalisation des fiches avant graphe.
- Tests unitaires sur config, cache et privacy.

Prochaines extensions :

- Calibrer empiriquement les seuils de projection, les pénalités de hubs et les tailles de nœuds sur plusieurs corpus.
- BERTopic pour des topics plus lisibles.
- Détection de stance par modèle local ou API.
- Annotation humaine assistée.
- Comparaison avec données Matomo agrégées.
- Graphe dynamique temporel des communautés discursives.
- Modèle markovien acteur-cadrage.
- Score agrégé de sortie de bulle.
- Détection de recodages adverses.

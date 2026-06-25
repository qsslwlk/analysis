# Post-traitement des graphes discursifs

Cette note documente les scripts d'exploration qui travaillent à partir des sorties V2.6.3 déjà produites. Ils ne rappellent pas le LLM et ne recalculent pas les embeddings : ils relisent les CSV du dossier `outputs/`, filtrent les relations trop bruitées, reconstruisent des graphes plus lisibles et exportent des tables d'audit.

## Pré-requis

Produire d'abord les fichiers discursifs V2.6.3 :

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --build-discourse-graph \
  --llm-provider ollama \
  --llm-model llama3.1:8b
```

Les scripts de post-traitement attendent au minimum :

- `outputs/discursive_units.csv`
- `outputs/discursive_nodes.csv`
- `outputs/discursive_edges.csv`
- `outputs/discursive_similarity_edges.csv`

Les visualisations HTML utilisent `pyvis`. Si le package n'est pas installé, les scripts produisent quand même les exports CSV, JSON et GEXF.

## Script principal : graphe post-traité

Commande recommandée :

```bash
python scripts/postprocess_discursive_graph.py \
  --input-dir outputs \
  --output-dir outputs/graph_postprocess \
  --include-similarity-edges \
  --similarity-threshold 0.85 \
  --drop-non-signal-labels
```

Ce script construit deux objets :

- un graphe complet d'audit, proche du graphe V2.6.3 original ;
- une projection commentaire-commentaire filtrée, utilisée pour recalculer des communautés plus lisibles.

La projection conserve par défaut les relations discursives suivantes :

- `COMMENT_HAS_FRAME`
- `COMMENT_HAS_CANONICAL_CLAIM`
- `COMMENT_HAS_ARGUMENT_FAMILY`
- `COMMENT_EXPRESSES_STANCE_TOWARD_TARGET`
- `COMMENT_TARGETS_ACTOR`
- `COMMENT_HAS_TONE`

Elle exclut volontairement les relations de structure comme source, vidéo, chaîne et période. Ces relations sont utiles pour l'audit, mais elles collent mécaniquement trop de commentaires ensemble dans une projection communautaire.

## Sorties du post-traitement

Le dossier `outputs/graph_postprocess/` contient :

- `postprocessed_comment_communities.csv` : une ligne par commentaire avec la communauté post-traitée.
- `postprocessed_community_summary.csv` : résumé des communautés, distributions d'acteurs, frames dominantes, exemples et citations.
- `postprocessed_comment_projection_edges.csv` : arêtes commentaire-commentaire de la projection filtrée.
- `postprocessed_attribute_contributions.csv` : attributs responsables des arêtes de projection.
- `postprocessed_graph_diagnostics.json` : tailles des graphes, composants, hubs, hubs non informatifs et paramètres du run.
- `discursive_full_graph.gexf` : graphe complet exportable dans Gephi.
- `discursive_comment_projection.gexf` : projection commentaire-commentaire exportable dans Gephi.
- `discursive_full_graph_by_source.html` : visualisation HTML du graphe complet.
- `discursive_comment_projection_by_source.html` : visualisation HTML de la projection.

Les HTML sont autonomes : les dépendances JavaScript de `pyvis` sont intégrées dans le fichier, ce qui évite de générer un dossier `lib/` à la racine du projet.

## Paramètres importants

- `--drop-non-signal-labels` retire des attributs comme `other`, `unknown`, `unclear`, `autre`, `non classé` et les nœuds marqués vagues.
- `--min-attribute-df` retire les attributs trop rares pour créer une arête robuste entre commentaires.
- `--max-attribute-df` retire les attributs trop fréquents, qui deviennent des hubs peu informatifs.
- `--include-similarity-edges` ajoute les arêtes de similarité hybride produites par la V2.6.3.
- `--similarity-threshold` contrôle la sélectivité des similarités directes.
- `--similarity-lambda` pondère les arêtes de similarité par rapport aux attributs partagés.
- `--relation-weights-json` permet de modifier les poids des types de relations sans éditer le code.
- `--viz-layout static` calcule un layout déterministe avec NetworkX et désactive la physique PyVis.
- `--full-graph-max-viz-nodes` et `--projection-max-viz-nodes` limitent la taille des HTML.

Exemple de pondération personnalisée :

```bash
python scripts/postprocess_discursive_graph.py \
  --input-dir outputs \
  --output-dir outputs/graph_postprocess_claims_first \
  --drop-non-signal-labels \
  --relation-weights-json '{"COMMENT_HAS_CANONICAL_CLAIM": 2.0, "COMMENT_HAS_TONE": 0.1}'
```

## Script RN/LFI : ponts discursifs

Commande recommandée :

```bash
python scripts/visualize_rn_lfi_bridges.py \
  --input-dir outputs \
  --output-dir outputs/rn_lfi_bridges \
  --source-a RN \
  --source-b LFI
```

Le script cherche deux formes de ponts :

- des similarités directes entre commentaires des deux sources ;
- des attributs discursifs partagés par des commentaires des deux sources.

Les sources sont configurables. Par exemple :

```bash
python scripts/visualize_rn_lfi_bridges.py \
  --input-dir outputs \
  --output-dir outputs/lfi_media_bridges \
  --source-a LFI \
  --source-b Media
```

## Sorties RN/LFI

Le dossier `outputs/rn_lfi_bridges/` contient :

- `rn_lfi_attribute_bridges.csv` : attributs partagés classés par score de pont.
- `rn_lfi_direct_similarity_bridges.csv` : arêtes de similarité directe entre commentaires des deux sources.
- `rn_lfi_bridge_subgraph_nodes.csv` : nœuds du sous-graphe de ponts.
- `rn_lfi_bridge_subgraph_edges.csv` : arêtes du sous-graphe de ponts.
- `rn_lfi_bridge_diagnostics.json` : paramètres, volumes et premiers ponts détectés.
- `rn_lfi_bridge_graph.gexf` : sous-graphe exportable dans Gephi.
- `rn_lfi_bridge_graph.html` : visualisation interactive.

## Rapport lisible des ponts

Le graphe sert à explorer, mais le livrable d'analyse doit être un rapport lisible. Le script `scripts/generate_bridge_report.py` transforme les ponts attributaires en fiches auditables avec score, citations et interprétation prudente.

Commande recommandée après `scripts/visualize_rn_lfi_bridges.py` :

```bash
python scripts/generate_bridge_report.py \
  --input-dir outputs \
  --bridge-dir outputs/rn_lfi_bridges \
  --output-dir outputs/rn_lfi_bridges
```

Le script produit :

- `bridge_report.csv` : table d'audit avec une ligne par pont.
- `bridge_report.md` : rapport lisible avec top ponts de contenu, top ponts de forme, fiches détaillées et grille d'audit.

Colonnes principales de `bridge_report.csv` :

- `bridge_label` : label du pont, par exemple `justice_fiscale` ou `hostile → macron`.
- `bridge_type` : famille lisible du nœud, par exemple `frame`, `claim`, `target`, `stance_target`, `tone`.
- `bridge_category` : `content`, `form` ou `mixed`.
- `n_RN` et `n_LFI` : nombre de commentaires reliés au pont pour chaque source.
- `balance_score` : équilibre entre les deux sources, de `0` à `1`.
- `specificity_score` : score IDF positif, plus élevé quand l'attribut est moins générique dans le corpus.
- `report_score` : score de classement du rapport.
- `top_RN_quotes` et `top_LFI_quotes` : citations et métadonnées au format JSON.
- `interpretation` : lecture automatique prudente, à vérifier par humain.
- `audit_label` et `audit_notes` : colonnes vides destinées à l'audit humain.

La formule du score de rapport est :

```text
report_score =
  log(1 + n_RN)
  × log(1 + n_LFI)
  × balance_score
  × specificity_score
  × type_weight
```

Avec :

```text
balance_score = 1 - abs(n_RN - n_LFI) / (n_RN + n_LFI)
specificity_score = log((N + 1) / (df + 1))
```

Ce score ne décide pas qu'un pont est vrai. Il sert seulement à ordonner les ponts à auditer. Le label final doit être renseigné dans `audit_label`.

## Score des ponts

Le score d'un attribut partagé favorise les ponts équilibrés et spécifiques :

```text
score =
  sqrt(poids_source_a * poids_source_b)
  × équilibre
  × poids_type_attribut
  × pénalité_hub
  × log1p(nombre_minimal_de_commentaires_par_source)
```

Les types les plus spécifiques sont favorisés :

- `StanceTargetNode`
- `CanonicalClaimNode`
- `FrameNode`

Par défaut, le script ne garde que `FrameNode`, `CanonicalClaimNode`, `TargetNode` et `StanceTargetNode`. Les types plus génériques restent disponibles via `--include-node-types`, mais ils sont pénalisés :

- `ArgumentFamilyNode`
- `ToneNode`
- `ThemeNode`

Cette pondération évite qu'un pont très large comme `programmatique`, `moral` ou `anger` domine mécaniquement des proximités discursives plus précises.

## Réglages utiles

Pour réduire le bruit :

```bash
python scripts/visualize_rn_lfi_bridges.py \
  --input-dir outputs \
  --output-dir outputs/rn_lfi_bridges_strict \
  --similarity-threshold 0.90 \
  --max-attribute-df 80 \
  --min-comments-per-side 3
```

Pour explorer plus largement :

```bash
python scripts/visualize_rn_lfi_bridges.py \
  --input-dir outputs \
  --output-dir outputs/rn_lfi_bridges_broad \
  --similarity-threshold 0.75 \
  --max-attribute-df 250 \
  --top-attribute-bridges 80 \
  --include-node-types FrameNode,CanonicalClaimNode,TargetNode,StanceTargetNode,ArgumentFamilyNode,ToneNode
```

Pour ne garder que les attributs les plus interprétables :

```bash
python scripts/visualize_rn_lfi_bridges.py \
  --input-dir outputs \
  --output-dir outputs/rn_lfi_bridges_specific \
  --include-node-types FrameNode,CanonicalClaimNode,StanceTargetNode
```

## Interprétation

Ces graphes servent à générer des hypothèses, pas à prouver des alignements idéologiques. Un pont signifie que deux ensembles de commentaires partagent une formulation, un cadrage, une cible ou une proximité sémantique dans le corpus collecté.

Avant de tirer une conclusion, vérifier :

- le nombre de commentaires par source ;
- la distribution des vidéos et des périodes ;
- les citations représentatives dans les tooltips ou les CSV ;
- la présence de hubs génériques dans `postprocessed_graph_diagnostics.json` ;
- la stabilité des ponts quand `--max-attribute-df`, `--similarity-threshold` et `--include-node-types` changent.

Un bon signal doit rester compréhensible à la lecture de quelques verbatims. Si le graphe montre un cluster ou un pont que les citations ne permettent pas d'expliquer, il faut le traiter comme du bruit ou comme un artefact de projection.

## Problèmes fréquents

Si les graphes restent trop denses :

- augmenter `--similarity-threshold` ;
- diminuer `--max-attribute-df` ;
- utiliser `--include-node-types FrameNode,CanonicalClaimNode,StanceTargetNode` ;
- réduire `--top-attribute-bridges` ;
- désactiver temporairement `--include-similarity-edges`.

Si les communautés sont trop fragmentées :

- diminuer `--similarity-threshold` ;
- augmenter `--max-attribute-df` ;
- diminuer `--min-attribute-df` ;
- diminuer `--min-community-size` ;
- tester une valeur plus basse de `--resolution`.

Si le HTML est lent :

- réduire `--full-graph-max-viz-nodes` ;
- réduire `--projection-max-viz-nodes` ;
- privilégier le GEXF dans Gephi pour les très gros graphes.

## Positionnement dans le pipeline

Ces scripts sont volontairement placés dans `scripts/` plutôt que dans le package `observatoire/`. Ils sont encore exploratoires : ils aident à calibrer les seuils, inspecter les hubs et choisir les visualisations utiles avant de stabiliser une API interne.

Une fois les paramètres validés sur plusieurs corpus, la logique pourra être déplacée dans un module package, avec tests unitaires dédiés sur :

- la déduplication des arêtes ;
- le filtrage des labels non informatifs ;
- la projection commentaire-commentaire ;
- le score des ponts entre sources ;
- la stabilité des diagnostics.

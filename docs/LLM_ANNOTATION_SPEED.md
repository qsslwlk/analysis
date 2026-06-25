# Accélérer l'annotation LLM locale

L'annotation des fiches discursives est l'étape la plus coûteuse du pipeline quand elle utilise un modèle local via Ollama. Avec `llama3.1:8b`, un run complet peut prendre plusieurs heures si chaque commentaire est annoté séquentiellement.

Cette note documente les leviers disponibles pour réduire ce temps sans perdre la traçabilité.

## Diagnostic

Le coût vient de quatre facteurs :

- un appel LLM par commentaire candidat ;
- un prompt riche qui demande thème, frame, stance, argument, tonalité, ambiguïtés et résumé ;
- une génération JSON contrainte ;
- un modèle 8B local qui peut être lent sur CPU ou GPU limité.

Le pipeline sépare maintenant deux étapes :

1. extraction/annotation discursive coûteuse ;
2. post-traitements rapides : graphes, seuils, ponts, rapports et visualisations.

## Ce qui est maintenant disponible

### Cache LLM local

Chaque réponse LLM de fiche discursive peut être stockée dans :

```text
outputs/discursive_card_llm_cache.jsonl
```

La clé de cache dépend :

- du prompt système ;
- du prompt utilisateur construit à partir du commentaire ;
- du namespace provider/modèle, par exemple `ollama:llama3.1:8b`.

Si le même commentaire est relancé avec le même prompt et le même modèle, le pipeline relit la réponse en cache au lieu de rappeler Ollama.

Pour désactiver ce cache :

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --disable-discursive-card-cache
```

Pour choisir un autre chemin :

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --discursive-card-cache outputs/cache/llama31_discourse_cards.jsonl
```

### Réutilisation de `discursive_cards.csv`

Si vous demandez seulement `--cluster-discourse-cards` ou `--build-discourse-graph`, le pipeline réutilise automatiquement `outputs/discursive_cards.csv` si ce fichier existe.

Relance rapide du graphe sans réannotation :

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --build-discourse-graph \
  --llm-provider ollama \
  --llm-model llama3.1:8b
```

Même si `--extract-discourse-cards` est présent, vous pouvez forcer la réutilisation :

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --build-discourse-graph \
  --reuse-discourse-cards
```

Pour refaire réellement l'annotation, ne passez pas `--reuse-discourse-cards`.

### Prompt rapide

Le prompt complet reste disponible :

```text
prompts/extract_discursive_card.md
```

Un prompt plus court existe pour les runs rapides :

```text
prompts/extract_discursive_card_fast.md
```

Commande :

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --build-discourse-graph \
  --llm-provider ollama \
  --llm-model llama3.1:8b \
  --discursive-card-prompt prompts/extract_discursive_card_fast.md
```

Compromis :

- plus rapide ;
- sortie plus concise ;
- moins d'ambiguïtés et de nuances secondaires ;
- souvent suffisante pour construire les nœuds principaux du graphe.

### Workers concurrents

Vous pouvez lancer plusieurs appels LLM en parallèle :

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --llm-provider ollama \
  --llm-model llama3.1:8b \
  --discursive-card-workers 2
```

Recommandation :

- commencer avec `--discursive-card-workers 2` ;
- tester `4` seulement si la machine a assez de RAM/GPU ;
- revenir à `1` si Ollama ralentit, swap ou devient instable.

Sur certaines machines, plusieurs workers accélèrent beaucoup. Sur d'autres, ils saturent le modèle et ralentissent tout. Il faut benchmarker localement.

### Options Ollama par environnement

Le client Ollama accepte ces variables :

```bash
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_NUM_PREDICT=768
export OLLAMA_NUM_CTX=2048
export OLLAMA_NUM_THREAD=8
```

Effets :

- `OLLAMA_KEEP_ALIVE` garde le modèle chargé plus longtemps.
- `OLLAMA_NUM_PREDICT` limite la longueur maximale de génération.
- `OLLAMA_NUM_CTX` limite le contexte.
- `OLLAMA_NUM_THREAD` peut améliorer le débit CPU selon la machine.

Valeurs prudentes pour commencer :

```bash
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_NUM_PREDICT=768
```

## Commande recommandée pour un premier run rapide

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

## Commande recommandée pour itérer après annotation

```bash
python -m observatoire.cli \
  --config config/corpus.example.json \
  --build-discourse-graph \
  --llm-provider ollama \
  --llm-model llama3.1:8b
```

Puis post-traiter sans LLM :

```bash
python scripts/postprocess_discursive_graph.py \
  --input-dir outputs \
  --output-dir outputs/graph_postprocess \
  --include-similarity-edges \
  --similarity-threshold 0.85 \
  --drop-non-signal-labels

python scripts/visualize_rn_lfi_bridges.py \
  --input-dir outputs \
  --output-dir outputs/rn_lfi_bridges

python scripts/generate_bridge_report.py \
  --input-dir outputs \
  --bridge-dir outputs/rn_lfi_bridges \
  --output-dir outputs/rn_lfi_bridges
```

## Benchmark conseillé

Avant un run de plusieurs heures, tester sur 50 commentaires :

```bash
time python -m observatoire.cli \
  --config config/corpus.example.json \
  --extract-discourse-cards \
  --llm-provider ollama \
  --llm-model llama3.1:8b \
  --discursive-card-limit 50 \
  --discursive-card-prompt prompts/extract_discursive_card_fast.md \
  --discursive-card-workers 2
```

Comparer ensuite :

- temps total ;
- nombre de fiches gardées dans `outputs/discursive_card_coverage.csv` ;
- qualité de quelques lignes dans `outputs/discursive_cards.csv` ;
- stabilité du graphe et des ponts.

Modèles à tester :

- `llama3.1:8b` : plus robuste, plus lent ;
- `mistral:7b` : souvent rapide et correct en français ;
- `qwen2.5:7b` : bon compromis possible ;
- `llama3.2:3b` ou `qwen2.5:3b` : beaucoup plus rapide, à auditer davantage.

## Ordre de priorité

1. Ne jamais relancer le LLM pour les graphes si `discursive_cards.csv` existe.
2. Utiliser le cache LLM par défaut.
3. Tester le prompt rapide.
4. Tester `--discursive-card-workers 2`.
5. Benchmarker un modèle plus petit sur 50 commentaires.
6. Garder le prompt complet pour les runs finaux si la nuance est plus importante que la vitesse.

## Limites

Le cache stocke les réponses LLM localement. Il peut contenir des extraits de commentaires via la réponse JSON. Il ne doit pas être publié sans vérifier la politique de données du projet.

Le prompt rapide peut lisser certaines nuances. Pour une publication ou une analyse sensible, auditer un échantillon et comparer quelques fiches produites par le prompt complet.

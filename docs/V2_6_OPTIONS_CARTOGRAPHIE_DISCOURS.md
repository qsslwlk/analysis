# V2.6 — Options techniques pour une cartographie robuste des discours

## Résumé Exécutif

La réduction des verbatims à quelques labels est insuffisante pour construire une cartographie méthodologiquement défendable des discours. Un commentaire politique peut porter simultanément un thème, un cadrage, une stance envers plusieurs cibles, un argument, une objection, une tonalité et une ambiguïté. Si le pipeline force chaque verbatim dans une catégorie unique, il produit une carte lisible mais appauvrie : les oppositions internes, les déplacements rhétoriques et les discours minoritaires disparaissent derrière des agrégats trop propres.

L'enjeu n'est donc pas seulement de mieux nommer des clusters. Il faut préserver la diversité discursive : distinguer ce dont parle le commentaire, comment il le cadre, contre qui ou quoi il se positionne, quel argument il mobilise, et avec quel niveau de certitude. Les verbatims doivent rester auditables, avec citations représentatives, afin que les catégories produites puissent être contestées, corrigées ou fusionnées par lecture humaine.

Trois options techniques sont envisageables. L'option 1, sans fine-tuning, consiste à utiliser un LLM généraliste pour produire une fiche discursive structurée par verbatim, puis à cartographier ces fiches par embeddings, clustering, matrices et graphes. C'est l'option la plus rapide et la plus défendable pour une V2.6, à condition d'ajouter des tests de stabilité et un audit humain. L'option 2, fine-tuning supervisé léger, devient pertinente si l'annotation structurée est instable, coûteuse ou trop dépendante du prompt. L'option 3, fine-tuning contrastif d'embeddings, est pertinente si l'objectif principal devient la qualité géométrique de la carte : proximités, oppositions, continuités et distances discursives.

La recommandation est progressive : commencer par l'option 1 comme baseline forte, mesurer ses limites, puis décider si un fine-tuning est justifié. Un fine-tuning brut sur les verbatims est à éviter : il risque surtout d'apprendre le style et la distribution moyenne du corpus, sans apprendre explicitement la structure analytique des discours.

## Comparaison Des Trois Options

### Option 1 — Pipeline Sans Fine-Tuning

#### Principe

Un LLM généraliste annote chaque verbatim avec une fiche discursive riche et structurée. La cartographie est ensuite construite à partir de ces fiches : embeddings des claims ou des résumés analytiques, clustering, BERTopic, UMAP/PCA, graphes de similarité, matrices acteur × cadrage × temps, et verbatims représentatifs.

L'objectif est d'utiliser le LLM comme outil d'extraction analytique, pas comme arbitre final. Chaque fiche doit rester ancrée dans le verbatim original et contenir des citations justificatives.

#### Données Nécessaires

- Verbatims publics nettoyés et anonymisés.
- Métadonnées agrégées : acteur, vidéo, date, séquence, titre.
- Prompt structuré avec schéma JSON stable.
- Petit échantillon humainement audité pour évaluer la qualité.
- Éventuellement quelques exemples few-shot validés.

#### Pipeline Technique

1. Filtrer les verbatims trop courts, réactionnels ou non discursifs.
2. Envoyer chaque verbatim au LLM avec métadonnées minimales.
3. Produire une fiche structurée :
   - thème principal ;
   - sous-thèmes ;
   - cadrage dominant ;
   - cadrages secondaires ;
   - stance par cible ;
   - argument central ;
   - type d'argument ;
   - objection ou attaque ;
   - tonalité ;
   - ambiguïtés ;
   - citations ;
   - confiance.
4. Valider le JSON, rejeter ou marquer les fiches invalides.
5. Embedding des claims, arguments, cadrages ou résumés discursifs.
6. Clustering et réduction dimensionnelle.
7. Production de cartes, matrices, temporalités et citations représentatives.
8. Audit humain d'un échantillon et rapport de limites.

#### Livrables Possibles

- `discursive_cards.csv` ou `discursive_cards.jsonl`.
- `discursive_clusters.csv`.
- `frame_actor_matrix.csv`.
- `stance_target_matrix.csv`.
- Carte UMAP/PCA des discours.
- Graphe de similarité entre familles discursives.
- Rapport Markdown avec citations représentatives.
- Tableau qualité : confiance, ambiguïtés, taux de fiches rejetées.

#### Coût Et Complexité

Complexité faible à moyenne. Le coût principal vient des appels LLM, de la latence et de la conception du prompt. L'architecture reste simple et facilement itérable. Avec Ollama, le coût monétaire peut être faible, mais la latence et la qualité varient selon le modèle local.

#### Avantages

- Mise en œuvre rapide.
- Pas besoin de corpus annoté massif.
- Forte expressivité analytique.
- Facile à auditer si les citations sont conservées.
- Compatible avec OpenAI ou Ollama.
- Bon choix pour tester plusieurs schémas d'analyse avant de figer une ontologie.

#### Limites

- Sensibilité au prompt.
- Variabilité entre modèles.
- Risque d'hallucination analytique si les preuves textuelles ne sont pas exigées.
- Coût potentiellement élevé sur gros corpus.
- Généralisation incertaine à de nouveaux contextes politiques ou formats de commentaires.

#### Risques Méthodologiques

- Surinterprétation de commentaires ambigus.
- Apparence de précision excessive via des champs structurés.
- Catégories trop dépendantes du prompt initial.
- Compression abusive si l'on n'autorise pas plusieurs cadrages ou stances.

#### Robustesse Attendue

Robustesse moyenne à bonne si trois conditions sont réunies : prompt stable, citations obligatoires, audit humain régulier. Sans ces garde-fous, la robustesse est seulement apparente.

#### Conditions De Pertinence

Cette option est pertinente pour V2.6 si l'objectif est d'obtenir rapidement une cartographie exploitable, de tester la valeur des fiches discursives et de construire des baselines avant tout fine-tuning.

### Option 2 — Fine-Tuning Supervisé Léger

#### Principe

Créer un échantillon annoté manuellement de verbatims avec fiches discursives riches, puis fine-tuner un modèle pour apprendre la tâche :

```text
verbatim + métadonnées → fiche discursive structurée
```

Le but n'est pas d'apprendre le style moyen du corpus, mais d'apprendre une opération d'analyse : distinguer thèmes, cadrages, stances, arguments, ambiguïtés, tonalités et formes de réception.

#### Données Nécessaires

- Échantillon annoté humainement, idéalement stratifié par acteur, période, volume et type de discours.
- Guide d'annotation stable.
- Double annotation sur une partie du corpus pour mesurer l'accord inter-annotateurs.
- Exemples négatifs et ambigus.
- Jeu de validation séparé, non utilisé pour l'entraînement.

#### Pipeline Technique

1. Définir un schéma de fiche discursive et un guide d'annotation.
2. Annoter quelques centaines à quelques milliers de verbatims.
3. Mesurer l'accord humain.
4. Nettoyer les désaccords ou les garder comme cas ambigus explicites.
5. Fine-tuner un modèle génératif ou instruction-tuned.
6. Évaluer sur jeu tenu à part.
7. Comparer avec option 1 zero-shot et few-shot.
8. Utiliser les fiches générées pour les mêmes cartes et matrices que l'option 1.

#### Livrables Possibles

- Guide d'annotation.
- Corpus annoté versionné.
- Modèle fine-tuné ou adaptateur léger.
- Rapport d'évaluation humain/modèle.
- Fiches discursives produites à grande échelle.
- Cartographie construite sur sorties supervisées.

#### Coût Et Complexité

Complexité moyenne à élevée. Le coût principal est humain : définir le guide, annoter, arbitrer les désaccords, maintenir la qualité. Le coût technique dépend du modèle retenu. Un fine-tuning léger peut être raisonnable, mais l'effort d'évaluation ne doit pas être sous-estimé.

#### Avantages

- Sorties plus stables si la tâche est bien spécifiée.
- Moins de dépendance au prompt.
- Possibilité d'adapter le modèle aux catégories analytiques du projet.
- Meilleure répétabilité pour des runs réguliers.
- Utile si les fiches doivent être produites à grande échelle.

#### Limites

- Nécessite un corpus annoté fiable.
- Risque de figer trop tôt une grille analytique.
- Maintenance nécessaire si les discours évoluent.
- Le modèle peut apprendre les biais des annotateurs.
- Moins flexible qu'un prompt pour explorer de nouveaux champs.

#### Risques Méthodologiques

- Naturaliser les catégories d'annotation comme si elles étaient objectives.
- Sous-représenter les discours rares dans l'échantillon annoté.
- Confondre stabilité du modèle et validité analytique.
- Réduire les ambiguïtés si le schéma force des choix uniques.

#### Robustesse Attendue

Robustesse bonne sur un domaine stable et bien annoté. Robustesse faible à moyenne hors distribution, notamment si de nouveaux acteurs, thèmes ou formats discursifs apparaissent.

#### Conditions De Pertinence

Cette option devient pertinente si l'option 1 montre une instabilité forte entre prompts ou modèles, ou si le volume à traiter rend nécessaire une annotation automatique plus standardisée.

### Option 3 — Fine-Tuning Contrastif D'Embeddings

#### Principe

Construire un modèle d'embedding spécialisé pour que la distance entre verbatims reflète la proximité discursive plutôt que la seule proximité lexicale. Le modèle apprend à rapprocher des verbatims qui partagent un cadrage ou une logique argumentative, et à éloigner des verbatims lexicalement proches mais discursivement opposés.

Exemples de contraintes :

- même cadrage, vocabulaire différent → proche ;
- même vocabulaire, positions opposées → éloigné ;
- même thème, cadrage différent → distance intermédiaire ;
- même argument, tonalité différente → proximité partielle.

#### Données Nécessaires

- Paires ou triplets annotés : ancre, positif, négatif.
- Labels discursifs ou jugements de similarité.
- Cas difficiles : opposition sur même vocabulaire, même thème avec cadrage différent, ironie, ambiguïté.
- Jeu de test de retrieval et de clustering.
- Échantillon humain pour évaluer la géométrie obtenue.

#### Pipeline Technique

1. Définir ce que signifie la proximité discursive.
2. Construire des paires ou triplets à partir d'annotations humaines et de hard negatives.
3. Fine-tuner un modèle SentenceTransformer ou équivalent avec perte contrastive/triplet.
4. Évaluer par retrieval : les voisins sont-ils discursivement proches ?
5. Produire UMAP/PCA, clusters, graphes et trajectoires dans l'espace spécialisé.
6. Comparer avec embeddings généralistes et TF-IDF.

#### Livrables Possibles

- Modèle d'embedding spécialisé.
- Jeu de paires/triplets versionné.
- Benchmark de retrieval discursif.
- Carte des proximités/oppositions.
- Graphe de similarité plus interprétable.
- Clusters fondés sur géométrie discursive plutôt que lexicale.

#### Coût Et Complexité

Complexité élevée. Cette option demande un design d'évaluation plus exigeant que les deux premières, car la qualité n'est pas seulement dans une fiche JSON mais dans la géométrie globale de l'espace. L'annotation de paires/triplets peut être plus difficile que l'annotation de fiches.

#### Avantages

- Meilleure cartographie des proximités et oppositions.
- Réduit le poids du vocabulaire de surface.
- Utile pour découvrir des familles discursives non évidentes lexicalement.
- Peut améliorer fortement la cohérence des clusters.

#### Limites

- Ne produit pas directement une fiche interprétable par verbatim.
- Nécessite souvent une couche de labeling ensuite.
- Plus difficile à expliquer à des non-spécialistes.
- Évaluation plus complexe.
- Peut masquer les raisons exactes d'une proximité si les citations ne sont pas liées à la carte.

#### Risques Méthodologiques

- Construire une géométrie séduisante mais peu auditable.
- Imposer une définition contestable de la proximité discursive.
- Perdre les discours rares si le sampling de paires/triplets est déséquilibré.
- Confondre séparation géométrique et séparation politique réelle.

#### Robustesse Attendue

Robustesse potentiellement élevée pour la cartographie si les paires/triplets sont bien conçus. Robustesse faible si le jeu contrastif est pauvre, trop lexical ou insuffisamment diversifié.

#### Conditions De Pertinence

Cette option est pertinente si l'objectif prioritaire devient la qualité de la carte elle-même : distances, oppositions, voisinages, trajectoires. Elle ne doit pas être le premier choix si l'on n'a pas encore stabilisé les catégories analytiques.

## Tableau Comparatif

| Option | Besoin En Données Annotées | Difficulté Technique | Capacité À Préserver La Nuance | Capacité À Généraliser | Risque De Compression Abusive | Facilité De Mise En Œuvre | Qualité Attendue De La Cartographie | Recommandation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Option 1 — LLM sans fine-tuning | Faible à moyen : audit humain + few-shot | Moyenne | Élevée si fiche multi-champs et citations | Moyenne, dépend du modèle et du prompt | Moyen | Élevée | Bonne baseline, perfectible | À faire en V2.6 |
| Option 2 — Fine-tuning supervisé léger | Moyen à élevé : fiches validées | Moyenne à élevée | Élevée si schéma riche | Bonne dans le domaine annoté | Moyen à élevé si schéma trop rigide | Moyenne | Bonne si annotation fiable | À envisager après mesure d'instabilité |
| Option 3 — Fine-tuning contrastif embeddings | Élevé : paires/triplets difficiles | Élevée | Moyenne directement, élevée avec citations et labels | Bonne si jeu contrastif diversifié | Faible à moyen pour la géométrie, mais risque d'opacité | Faible | Potentiellement très bonne pour proximités/oppositions | À réserver si la carte devient l'objectif principal |

## Métriques D'Évaluation

### Accord Humain / Modèle

- Comparer les champs structurés du modèle avec un échantillon annoté humainement.
- Mesurer l'accord par champ : thème, cadrage, stance, argument, tonalité, ambiguïté.
- Ne pas chercher uniquement un score global : les erreurs de stance ou de cadrage sont plus graves que les variations de formulation.

### Stabilité Entre Prompts

- Exécuter plusieurs variantes du prompt sur le même échantillon.
- Mesurer la stabilité des fiches, des clusters et des labels finaux.
- Identifier les champs les plus sensibles.

### Stabilité Entre Modèles

- Comparer OpenAI, Ollama et éventuellement plusieurs modèles open-source.
- Mesurer les divergences sur les cas ambigus.
- Conserver les cas de désaccord comme matériau d'audit, pas seulement comme erreurs.

### Diversité Des Cadrages Détectés

- Compter les cadrages distincts après normalisation.
- Vérifier que la diversité ne vient pas seulement de formulations synonymes.
- Repérer si le pipeline reconduit toujours les mêmes catégories dominantes.

### Couverture Des Discours Minoritaires

- Mesurer le taux de verbatims rares absorbés par des clusters majoritaires.
- Suivre la part de bruit `-1` et les petits clusters.
- Lire manuellement les clusters minoritaires avant de les supprimer.

### Cohérence Des Clusters

- Mesures internes : silhouette, Davies-Bouldin, distance intra/inter-cluster.
- Mesures qualitatives : cohérence des citations proches du centroïde.
- Test de stabilité par bootstrap : les mêmes familles réapparaissent-elles ?

### Fidélité Aux Verbatims

- Chaque fiche doit contenir au moins une citation représentative.
- Vérifier que le cadrage et la stance sont justifiés par le texte.
- Mesurer le taux de claims sans preuve textuelle suffisante.

### Citations Représentatives

- Évaluer si les citations choisies expliquent vraiment le cluster.
- Comparer citations proches du centroïde et citations limites.
- Documenter les contre-exemples et le bruit.

### Robustesse Temporelle

- Tester le pipeline sur plusieurs périodes.
- Vérifier si les catégories restent stables sans empêcher l'émergence de nouveaux discours.
- Repérer les ruptures artificielles dues au prompt ou au modèle.

### Comparaison Aux Baselines Lexicales

- Comparer avec TF-IDF + clustering.
- Comparer avec embeddings généralistes + clustering.
- Vérifier que le pipeline LLM ajoute de la valeur : nuance, couverture, citations, cohérence.

## Baselines Minimales Avant Fine-Tuning

Avant de justifier un fine-tuning, il faut établir au minimum les baselines suivantes :

1. Annotation LLM zero-shot structurée.
2. Annotation LLM few-shot avec exemples validés.
3. BERTopic sans LLM.
4. Embeddings généralistes + clustering.
5. TF-IDF + clustering.
6. Petit échantillon annoté humainement.

Ces baselines doivent être comparées sur le même échantillon, avec les mêmes métriques et une lecture humaine des erreurs. Le fine-tuning n'est justifié que s'il améliore clairement la stabilité, la fidélité aux verbatims ou la qualité de la géométrie discursive.

## Recommandation Finale

Pour V2.6, la trajectoire la plus défendable est de commencer par l'option 1 : annotation structurée par LLM, citations obligatoires, clustering des fiches discursives, matrices acteur × cadrage × temps et audit humain. Cette option permet de tester rapidement le schéma d'analyse sans figer prématurément une taxonomie.

L'option 2 ne doit être engagée que si les sorties de l'option 1 sont trop instables, trop coûteuses ou trop dépendantes du prompt. Le fine-tuning supervisé doit apprendre une tâche d'analyse, pas la distribution moyenne des commentaires.

L'option 3 ne doit être engagée que si la priorité devient la qualité de la géométrie discursive : obtenir une carte où les distances représentent vraiment des proximités, oppositions et continuités discursives. Elle est puissante, mais elle demande un investissement méthodologique plus lourd.

Le fine-tuning brut sur les verbatims est à éviter. Il risque surtout d'apprendre le style, les tics lexicaux, les acteurs dominants et la distribution moyenne du corpus. Il ne garantit pas que le modèle comprenne mieux la structure des discours.

## Garde-Fous Politiques, Juridiques Et Méthodologiques

- Ne jamais présenter les commentaires publics comme représentatifs de l'électorat.
- Parler de circulation des discours, pas d'opinion générale.
- Produire uniquement des analyses agrégées.
- Ne pas faire de scoring individuel.
- Ne pas exploiter de données sensibles.
- Conserver les verbatims originaux, anonymisés, pour auditabilité.
- Distinguer thème, cadrage, stance, argument et discours.
- Éviter les promesses causales fortes.
- Valider humainement les catégories critiques.
- Documenter les prompts, modèles, versions et paramètres.
- Signaler les volumes faibles et les clusters instables.
- Préserver les cas ambigus au lieu de les forcer dans une catégorie unique.

## Proposition De Prototype En 3 À 4 Semaines

### Semaine 1 — Socle De Données Et Fiches

- Collecter quelques centaines ou milliers de verbatims publics.
- Nettoyer, anonymiser et filtrer les verbatims non discursifs.
- Définir le schéma `discursive_card`.
- Implémenter l'annotation structurée par LLM.
- Produire un premier fichier `discursive_cards.jsonl`.

### Semaine 2 — Cartographie Et Baselines

- Générer embeddings sur claims, arguments et résumés discursifs.
- Produire clustering, UMAP/PCA et graphes de similarité.
- Produire matrices acteur × cadrage et acteur × stance.
- Lancer baselines TF-IDF, embeddings généralistes et BERTopic.
- Comparer les premiers résultats.

### Semaine 3 — Audit Et Stabilisation

- Auditer humainement un échantillon stratifié.
- Mesurer stabilité entre prompts et modèles.
- Identifier les champs les plus fragiles.
- Ajuster le schéma et les prompts.
- Ajouter des citations représentatives et contre-exemples par cluster.

### Semaine 4 — Rapport Et Décision

- Produire un rapport final avec limites.
- Présenter la carte des cadrages et temporalités simples.
- Documenter les discours minoritaires et les zones de bruit.
- Décider si un fine-tuning supervisé ou contrastif est justifié.
- Formaliser les garde-fous pour les versions suivantes.

## Décision Proposée Pour V2.6

La V2.6 devrait implémenter une version contrôlée de l'option 1 :

- fiche discursive structurée par verbatim ;
- preuve textuelle obligatoire ;
- conservation de l'ambiguïté ;
- clustering sur représentations discursives, pas seulement sur verbatims bruts ;
- comparaison à des baselines simples ;
- audit humain minimal ;
- rapport de stabilité et de limites.

Cette décision maximise l'apprentissage méthodologique tout en limitant le coût et le risque de sur-ingénierie. Elle laisse ouvertes les options 2 et 3, mais exige des preuves empiriques avant d'engager un fine-tuning.

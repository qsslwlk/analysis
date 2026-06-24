# V2.6.3 — Hypergraphe discursif typé

## 1. Résumé Exécutif

Le pipeline actuel `commentaire → embedding → clustering` donne une carte trop fragile pour analyser des discours publics. Les embeddings capturent des proximités utiles, mais ils mélangent souvent plusieurs dimensions : vocabulaire, noms propres, style militant, longueur du commentaire, source vidéo, tonalité émotionnelle et répétition de slogans. Deux commentaires peuvent être proches lexicalement sans porter la même configuration discursive ; inversement, deux commentaires peuvent partager un cadrage politique très proche tout en utilisant des formulations différentes.

La V2.6.3 propose de représenter chaque commentaire comme une configuration relationnelle : une source, une vidéo, une période, des thèmes, des frames, des claims, des cibles, des stances, des familles argumentatives, des tonalités et des registres rhétoriques. Cette configuration devient un hypergraphe discursif typé. Une communauté discursive n'est alors plus un simple paquet de phrases similaires, mais un sous-graphe relativement stable reliant certains cadrages, arguments, cibles, stances, sources et périodes.

L'objectif n'est pas de remplacer les embeddings, mais de les remettre à leur place : un signal parmi d'autres. La similarité entre commentaires doit être construite à partir des relations discursives extraites, pondérées, auditables et contextualisées. Les embeddings servent en complément pour capter des proximités sémantiques non couvertes par le schéma.

La recommandation opérationnelle est de commencer par une approche interprétable sans GNN : extraction LLM structurée, construction d'un graphe biparti ou d'un hypergraphe léger, pondération IDF, détection de communautés avec Leiden/Louvain ou clustering spectral, puis validation humaine. Le message passing neuronal ne doit venir qu'ensuite, si le volume, la stabilité des annotations et une tâche supervisée claire le justifient.

## 2. Définition Des Objets

### Objets discursifs

- **Commentaire** : verbatim public collecté sous une vidéo ou un contenu source. Il est l'unité d'observation minimale, mais pas nécessairement l'unité argumentative minimale.
- **Segment discursif** : fragment cohérent d'un commentaire portant une idée, un cadrage, une stance ou une objection identifiable. Un commentaire long peut contenir plusieurs segments.
- **Thème** : sujet général abordé, par exemple pouvoir d'achat, médias, immigration, institutions, guerre, écologie. Le thème dit de quoi parle le commentaire, pas comment il en parle.
- **Frame** : cadrage interprétatif appliqué au thème, par exemple “biais médiatique”, “élite contre peuple”, “compétence économique”, “trahison politique”. Le frame décrit la grille de lecture mobilisée.
- **Stance** : orientation du commentaire envers une cible donnée : `supportive`, `hostile`, `ambivalent` ou `unclear`. La stance doit toujours être conditionnée par une cible.
- **Cible** : entité vers laquelle une stance, une critique, une défense ou une accusation est dirigée. Cela peut être une personne, un parti, une institution, un média, un groupe social ou une catégorie abstraite.
- **Claim brut** : assertion extraite au plus près du verbatim, avec sa formulation encore dépendante du texte original.
- **Claim canonique** : reformulation normalisée d'un claim brut, réduisant les variantes lexicales sans ajouter d'interprétation non soutenue par le texte.
- **Famille argumentative** : type d'argument mobilisé, par exemple compétence, crédibilité, injustice, souveraineté, pouvoir d'achat, corruption, sécurité, biais médiatique.
- **Tonalité émotionnelle** : affect dominant exprimé ou performé par le commentaire, par exemple indignation, admiration, ironie, colère, méfiance, enthousiasme.
- **Registre rhétorique** : forme d'expression utilisée, par exemple soutien partisan, dénonciation, sarcasme, témoignage personnel, appel au vote, accusation, fact-checking, moquerie.
- **Source_actor** : acteur politique, médiatique ou institutionnel associé au contenu source. Il ne doit pas être confondu avec l'auteur du commentaire.
- **Vidéo source** : contenu public sous lequel le commentaire a été publié. Elle fournit le contexte de réception.
- **Période temporelle** : bucket temporel utilisé pour l'analyse dynamique, par exemple jour, semaine, séquence électorale ou période de campagne.
- **Communauté discursive** : sous-graphe stable reliant des commentaires à des combinaisons récurrentes de frames, claims, cibles, stances, familles argumentatives, sources et périodes.

### Distinctions critiques

- **Acteur qui publie ou porte la vidéo** : entité associée au contenu source, par exemple une chaîne, un média, un parti ou une personnalité invitée.
- **Acteur mentionné** : entité citée dans le commentaire, sans être nécessairement la cible d'une stance.
- **Cible de la stance** : entité envers laquelle le commentaire exprime une orientation favorable, hostile, ambivalente ou floue.
- **Position du commentateur** : posture discursive inférée à partir du commentaire, à ne pas convertir en opinion personnelle stable ni en attribut sensible.
- **Cadrage du commentaire** : manière dont le commentaire donne sens à la situation, indépendamment de la personne ou du parti qu'il soutient ou attaque.

Cette séparation est essentielle. Un commentaire sous une vidéo de Bardella peut mentionner Mélenchon, attaquer une journaliste, défendre Bardella et mobiliser un frame de biais médiatique. Réduire cela à “pro-Bardella” ou “cluster Bardella” détruit la structure discursive utile.

## 3. Schéma De Données

Chaque commentaire candidat est transformé par extraction LLM en objet JSON validable. Le schéma doit être stable, versionné et conservateur : quand le texte ne permet pas de trancher, le modèle doit produire `unclear` ou ajouter une incertitude plutôt que remplir artificiellement les champs.

```json
{
  "comment_id": "comment_123",
  "source_video_id": "video_456",
  "source_video_title": "Titre de la vidéo source",
  "source_actor": "acteur associé à la vidéo",
  "time_bucket": "2026-W25",
  "raw_text": "Texte original du commentaire",
  "canonical_rewrite": "Reformulation courte et neutre du contenu discursif",
  "themes": [
    "médias",
    "campagne électorale"
  ],
  "frames": [
    "biais médiatique",
    "leader crédible face à une attaque"
  ],
  "targets": [
    "Jordan Bardella",
    "journaliste"
  ],
  "stances": [
    {
      "target": "Jordan Bardella",
      "stance": "supportive",
      "confidence": 0.86,
      "evidence_quote": "il répond clairement"
    },
    {
      "target": "journaliste",
      "stance": "hostile",
      "confidence": 0.82,
      "evidence_quote": "la journaliste essaye encore de le piéger"
    }
  ],
  "claims_raw": [
    "la journaliste essaye de le piéger",
    "Bardella répond clairement"
  ],
  "claims_canonical": [
    "le traitement médiatique est perçu comme hostile ou biaisé",
    "le leader soutenu est perçu comme clair et crédible"
  ],
  "argument_families": [
    "crédibilité du leader",
    "biais médiatique"
  ],
  "rhetorical_tone": [
    "soutien partisan",
    "dénonciation"
  ],
  "emotion": [
    "indignation",
    "admiration"
  ],
  "register": [
    "commentaire militant",
    "critique des médias"
  ],
  "uncertainties": [
    "la portée exacte de l'accusation envers la journaliste reste implicite"
  ],
  "extraction_confidence": 0.81,
  "prompt_version": "discursive-hypergraph-v1",
  "model_version": "llama3.1:8b"
}
```

Contraintes de qualité :

- `stances` doit être une liste d'objets cible-orientation-preuve, jamais un label global unique.
- `claims_canonical` doit rester ancré dans `claims_raw` et ne pas généraliser au-delà du texte.
- `canonical_rewrite` doit reformuler le contenu discursif, pas le rendre plus intelligent qu'il ne l'est.
- `evidence_quote` doit contenir une citation courte présente dans le commentaire.
- `uncertainties` doit être utilisé activement pour les commentaires ironiques, elliptiques ou ambigus.
- `prompt_version` et `model_version` sont obligatoires pour comparer la stabilité entre prompts et modèles.

## 4. Construction De L'Hypergraphe

### Types de nœuds

- **CommentNode** : commentaire anonymisé ou identifiant interne.
- **VideoNode** : vidéo source.
- **SourceActorNode** : acteur associé au contenu source.
- **ThemeNode** : thème général.
- **FrameNode** : cadrage interprétatif.
- **ClaimNode** : claim brut.
- **CanonicalClaimNode** : claim normalisé.
- **ArgumentFamilyNode** : famille argumentative.
- **TargetNode** : acteur ou entité ciblée.
- **StanceTargetNode** : paire `(target, stance)`, par exemple `(journaliste, hostile)`.
- **ToneNode** : tonalité émotionnelle ou rhétorique.
- **RegisterNode** : registre discursif.
- **TimeNode** : bucket temporel.
- **ChannelNode** : chaîne, média, plateforme ou canal source si disponible.

### Types de relations

- **COMMENT_HAS_FRAME** : commentaire lié à un frame extrait.
- **COMMENT_HAS_CLAIM** : commentaire lié à un claim brut.
- **COMMENT_HAS_CANONICAL_CLAIM** : commentaire lié à un claim canonique.
- **COMMENT_HAS_ARGUMENT_FAMILY** : commentaire lié à une famille argumentative.
- **COMMENT_TARGETS_ACTOR** : commentaire mentionne ou cible un acteur.
- **COMMENT_EXPRESSES_STANCE_TOWARD_TARGET** : commentaire exprime une stance envers une cible.
- **COMMENT_HAS_TONE** : commentaire porte une tonalité.
- **COMMENT_HAS_REGISTER** : commentaire mobilise un registre rhétorique.
- **COMMENT_FROM_VIDEO** : commentaire publié sous une vidéo donnée.
- **VIDEO_ASSOCIATED_WITH_SOURCE_ACTOR** : vidéo associée à un acteur source.
- **COMMENT_IN_TIME_BUCKET** : commentaire assigné à une période.
- **CLAIM_BELONGS_TO_ARGUMENT_FAMILY** : claim canonique rattaché à une famille argumentative.
- **FRAME_CO_OCCURS_WITH_CLAIM** : frame et claim co-occurent dans un ou plusieurs commentaires.

### Option A — Hypergraphe natif

Chaque commentaire devient une hyperarête reliant simultanément plusieurs nœuds typés :

```text
comment_hyperedge_i = {
  CommentNode_i,
  VideoNode_j,
  SourceActorNode_k,
  TimeNode_t,
  FrameNode_a,
  CanonicalClaimNode_b,
  TargetNode_c,
  StanceTargetNode_d,
  ArgumentFamilyNode_e,
  ToneNode_f,
  RegisterNode_g
}
```

Cette représentation respecte mieux la nature relationnelle du discours : le sens vient de la co-présence conditionnelle des éléments, pas seulement de relations binaires isolées. Elle est utile pour des méthodes d'hypergraph spectral clustering ou des librairies comme HyperNetX.

Limite : l'outillage produit, visualisation et débogage est moins mature qu'avec des graphes classiques. Pour un POC, l'hypergraphe natif est intéressant conceptuellement, mais peut ralentir l'itération.

### Option B — Graphe biparti commentaire-attribut

On construit un graphe biparti avec, d'un côté, les commentaires et, de l'autre, les attributs discursifs typés. Les relations sont binaires, pondérées et typées :

```text
CommentNode_i --COMMENT_HAS_FRAME--> FrameNode_a
CommentNode_i --COMMENT_HAS_CANONICAL_CLAIM--> CanonicalClaimNode_b
CommentNode_i --COMMENT_EXPRESSES_STANCE_TOWARD_TARGET--> StanceTargetNode_d
CommentNode_i --COMMENT_FROM_VIDEO--> VideoNode_j
```

Cette option est plus simple à implémenter avec NetworkX, igraph, scipy sparse et Leiden/Louvain. Elle permet de calculer des matrices d'incidence par type de relation, de projeter le graphe sur les commentaires, puis de détecter des communautés discursives.

Recommandation POC : commencer par le graphe biparti typé, tout en gardant le vocabulaire d'hypergraphe dans le modèle conceptuel. Le graphe biparti est une approximation pratique de l'hypergraphe.

## 5. Pondération Des Relations

Sans pondération, les nœuds génériques dominent le graphe. Des termes comme “France”, “peuple”, “politique”, “colère”, “Macron”, “journaliste” ou “médias” peuvent devenir des hubs qui relient artificiellement des commentaires très différents.

### Formule de base

Pour une relation entre un commentaire `c` et un nœud discursif `n` :

```text
w(c, n) = relation_weight(r) * confidence(c, n) * log(N / df(n))
```

où :

- `r` est le type de relation ;
- `N` est le nombre total de commentaires ;
- `df(n)` est le nombre de commentaires reliés au nœud `n` ;
- `relation_weight(r)` encode l'importance analytique de la relation ;
- `confidence(c, n)` vient du score LLM ou d'une règle de validation.

Version stabilisée :

```text
w(c, n) = relation_weight(r) * confidence(c, n) * log((N + 1) / (df(n) + 1))
```

On peut ensuite normaliser les poids par commentaire pour éviter qu'un commentaire très riche écrase les autres.

### Poids par relation

Poids initiaux proposés, à calibrer empiriquement :

| Relation | Poids initial | Raison |
| --- | ---: | --- |
| `COMMENT_HAS_FRAME` | 1.40 | Le frame structure fortement le discours. |
| `COMMENT_HAS_CANONICAL_CLAIM` | 1.30 | Les claims canoniques rapprochent les formulations équivalentes. |
| `COMMENT_EXPRESSES_STANCE_TOWARD_TARGET` | 1.25 | La paire stance-cible est très informative. |
| `COMMENT_HAS_ARGUMENT_FAMILY` | 1.10 | Utile mais parfois plus générique que le claim. |
| `COMMENT_TARGETS_ACTOR` | 0.90 | Les noms propres peuvent dominer artificiellement. |
| `COMMENT_HAS_TONE` | 0.60 | La tonalité rapproche souvent le style plus que le fond. |
| `COMMENT_HAS_REGISTER` | 0.55 | Le registre est utile en second ordre. |
| `COMMENT_FROM_VIDEO` | 0.35 | La vidéo explique le contexte mais ne doit pas définir le discours. |
| `COMMENT_IN_TIME_BUCKET` | 0.25 | La temporalité sert surtout aux analyses dynamiques. |

### Confiance et preuves

- Une extraction sans `evidence_quote` reçoit un malus fort ou est rejetée.
- Une relation avec `confidence < 0.50` peut être conservée pour audit, mais exclue du graphe principal.
- Les claims et stances avec citation exacte sont favorisés.
- Les commentaires ironiques ou elliptiques doivent être marqués comme incertains plutôt que forcés.

### Pénalisation des nœuds vagues

Créer une liste de nœuds vagues ou trop génériques par type :

- thèmes génériques : “politique”, “France”, “société” ;
- émotions génériques : “colère”, “peur”, “soutien” ;
- cibles trop larges : “les gens”, “le peuple”, “les Français” ;
- claims pauvres : “X a raison”, “Y ment”, “il faut voter”.

Ces nœuds peuvent être :

- exclus des projections de similarité ;
- conservés seulement comme contexte ;
- plafonnés par un poids maximal ;
- fusionnés dans une catégorie `generic_or_low_information`.

### Noms propres vs frames

Les noms propres ne doivent pas jouer le même rôle que les frames. Deux commentaires qui mentionnent “Bardella” ne partagent pas nécessairement le même discours. Deux commentaires qui mobilisent le frame “biais médiatique” contre une cible journalistique ont une proximité discursive plus forte, même s'ils citent des acteurs différents.

Recommandation : séparer explicitement :

- `TargetNode` pour les acteurs ;
- `StanceTargetNode` pour la relation stance-cible ;
- `FrameNode` pour le cadrage ;
- `CanonicalClaimNode` pour l'assertion normalisée.

## 6. Similarité Entre Commentaires

À partir du graphe biparti, on construit des matrices d'incidence pondérées par type de relation :

- `B_frame` connecte commentaires et frames ;
- `B_claim` connecte commentaires et claims canoniques ;
- `B_stance` connecte commentaires et paires stance-cible ;
- `B_target` connecte commentaires et cibles ;
- `B_argument` connecte commentaires et familles argumentatives ;
- `B_tone` connecte commentaires et tonalités ;
- `E` contient les embeddings du texte original ou de la reformulation canonique.

La similarité hybride entre commentaires peut être définie par :

```text
K =
  λ_frame    B_frame    B_frame^T
+ λ_claim    B_claim    B_claim^T
+ λ_stance   B_stance   B_stance^T
+ λ_target   B_target   B_target^T
+ λ_argument B_argument B_argument^T
+ λ_tone     B_tone     B_tone^T
+ λ_embed    E          E^T
```

Chaque matrice doit être normalisée, par exemple en cosine similarity sur les lignes, afin qu'un type de relation très dense ne domine pas mécaniquement.

Poids initiaux proposés :

| Composante | Poids initial |
| --- | ---: |
| `λ_frame` | 0.25 |
| `λ_claim` | 0.25 |
| `λ_stance` | 0.20 |
| `λ_argument` | 0.12 |
| `λ_target` | 0.08 |
| `λ_tone` | 0.04 |
| `λ_embed` | 0.06 |

Ces poids ne sont pas des vérités théoriques. Ils doivent être testés par stabilité des communautés, audit qualitatif et comparaison à des labels humains.

### Pourquoi c'est préférable à une simple similarité cosine

Une similarité cosine sur embeddings a tendance à confondre :

- proximité lexicale et proximité argumentative ;
- noms propres et stances ;
- ton militant et contenu discursif ;
- commentaires du même contexte vidéo et discours réellement partagés ;
- slogans courts et claims structurés.

La similarité hybride force le modèle à expliciter ce qui rapproche deux commentaires. Si deux commentaires sont proches, on peut inspecter les frames, claims, stances et cibles responsables de cette proximité. La carte devient contestable et auditable, ce qui est indispensable pour une analyse politique prudente.

## 7. Détection De Communautés Discursives

### Méthodes recommandées en POC

1. **Leiden ou Louvain sur graphe pondéré**
   - Construire une projection commentaire-commentaire à partir de `K`.
   - Garder les arêtes au-dessus d'un seuil ou les `k` plus proches voisins.
   - Détecter des communautés pondérées.
   - Avantage : rapide, interprétable, robuste sur graphes clairsemés.

2. **Clustering spectral sur `K`**
   - Utiliser la matrice de similarité hybride.
   - Tester plusieurs nombres de clusters.
   - Avantage : bon contrôle expérimental.
   - Limite : nécessite de choisir ou estimer `k`.

3. **HDBSCAN sur embeddings de graphe**
   - Générer une représentation vectorielle des commentaires à partir des incidences pondérées.
   - Laisser HDBSCAN produire du bruit plutôt que forcer tous les commentaires dans une communauté.
   - Avantage : compatible avec des volumes modestes et des outliers.

### Méthodes avancées

4. **Node2Vec ou metapath2vec**
   - Apprendre des embeddings de nœuds à partir de marches aléatoires typées.
   - Utile si le graphe devient suffisamment grand et hétérogène.

5. **Hypergraph spectral clustering**
   - Utiliser directement l'hypergraphe commentaire-attribut.
   - Plus fidèle conceptuellement, mais outillage plus expérimental.

6. **Hypergraph Neural Network ou Heterogeneous GNN**
   - Apprendre par message passing sur nœuds et relations typées.
   - À réserver à une phase avancée avec volume important, labels de validation et objectif supervisé clair.

### Pourquoi commencer non-neuronal

Les méthodes non-neuronales permettent de répondre à trois questions avant de complexifier :

- Les extractions LLM sont-elles assez stables ?
- Les communautés trouvées sont-elles interprétables ?
- Les proximités sont-elles explicables par des relations auditables ?

Un GNN peut améliorer une tâche supervisée, mais il peut aussi masquer les erreurs d'ontologie, amplifier les hubs et rendre la carte plus séduisante que fiable.

## 8. Message Passing

### Intuition

Dans un graphe discursif, un commentaire reçoit de l'information des frames, claims, cibles, stances, sources et périodes auxquels il est connecté. Ces nœuds reçoivent eux-mêmes de l'information des autres commentaires qui les partagent. Le message passing permet donc de propager une représentation entre commentaires indirectement reliés par des objets discursifs communs.

Un commentaire qui partage avec d'autres le frame “biais médiatique”, le claim canonique “le traitement journalistique est hostile” et la stance `(journaliste, hostile)` peut être rapproché d'eux même si ses mots exacts sont différents.

### Formulation simple

Pour un nœud `v` à la couche `l` :

```text
h_v^{l+1} = σ(
  W_self h_v^l
  + Σ_r Σ_{u ∈ N_r(v)} α_{uv}^{r} W_r h_u^l
)
```

où :

- `r` est le type de relation ;
- `N_r(v)` est le voisinage de `v` selon la relation `r` ;
- `W_r` est une matrice propre au type de relation ;
- `α_{uv}^{r}` est un poids d'attention ou de compatibilité ;
- `σ` est une non-linéarité ;
- `h_v^l` est la représentation du nœud à la couche `l`.

### Risques

- **Propagation d'erreurs LLM** : une mauvaise extraction de frame ou de stance contamine plusieurs représentations.
- **Sur-lissage** : les commentaires deviennent trop similaires après plusieurs couches.
- **Domination des hubs** : des nœuds fréquents comme “France”, “Macron” ou “médias” attirent trop de messages.
- **Fuite temporelle** : si le graphe complet est utilisé, les commentaires futurs peuvent influencer les représentations passées.
- **Interprétabilité produit** : il devient plus difficile d'expliquer pourquoi une communauté existe.

Recommandation : ne pas utiliser de message passing profond dans la première V2.6.3. Si une version neuronale est testée, limiter le nombre de couches, contrôler les hubs, respecter les splits temporels et comparer systématiquement avec la version interprétable.

## 9. Analyse Temporelle

La dimension temporelle doit être native, pas ajoutée après coup. On construit un graphe par période `G_t`, ou un graphe global avec nœuds `TimeNode` et vues filtrées par période.

### Principes

- Construire les représentations d'une période avec les données disponibles jusqu'à cette période.
- Éviter que les commentaires futurs influencent les clusters passés.
- Mesurer la naissance, la persistance, la diffusion et la disparition des configurations discursives.
- Séparer volume, centralité et diversité : un discours peut être très volumineux mais peu transversal.

### Métriques

- **Persistance d'un claim** : nombre de périodes où un claim canonique reste actif au-dessus d'un seuil.
- **Diversité des sources d'un claim** : nombre et dispersion des `source_actor` associés à ce claim.
- **Centralité temporelle d'un frame** : centralité du frame dans chaque `G_t`, puis trajectoire.
- **Émergence d'une attaque** : apparition rapide d'une paire target-stance hostile ou d'un claim accusatoire.
- **Franchissement de bulle** : passage d'un frame ou claim entre communautés, sources ou acteurs distincts.
- **Déplacement de réception** : changement des stances associées à une même cible selon les périodes ou sources.
- **Stabilité d'une communauté discursive** : similarité Jaccard des nœuds structurants entre `G_t` et `G_{t+1}`.

### Sorties temporelles utiles

- trajectoires des frames par semaine ;
- heatmap `source_actor × frame × temps` ;
- heatmap `target × stance × temps` ;
- courbes de persistance des claims ;
- timeline des attaques ou controverses ;
- détection des frames émergents.

## 10. Sorties Produit

La V2.6.3 doit produire des sorties interprétables, pas seulement des clusters.

### Cartographie

- Carte des communautés discursives avec taille, cohésion et proportion de bruit.
- Top frames par communauté.
- Top claims canoniques par communauté.
- Top cibles et acteurs mentionnés.
- Top paires `stance-target`.
- Nœuds-ponts entre communautés.
- Commentaires représentatifs et citations courtes.

### Tableaux analytiques

- Matrice `source_actor × frame × temps`.
- Matrice `target × stance × temps`.
- Matrice `argument_family × source_actor`.
- Matrice `claim_canonical × time_bucket`.
- Tableau des nœuds génériques pénalisés ou exclus.
- Tableau de couverture : commentaires extraits, rejetés, ambigus, outliers.

### Scores opérationnels

- **Score de persistance d'une attaque** : durée et intensité d'un claim hostile envers une cible.
- **Score de transversalité d'un claim** : diversité des sources ou communautés qui l'emploient.
- **Score de cohésion communautaire** : densité interne pondérée par relations discursives.
- **Score de fragilité** : part de relations à faible confiance dans une communauté.
- **Score de nouveauté** : distance d'un nouveau sous-graphe aux configurations passées.

### Rapport hebdomadaire

Un rapport interprétable peut contenir :

- les communautés émergentes ;
- les frames en hausse ;
- les claims persistants ;
- les attaques ou accusations nouvelles ;
- les déplacements de réception ;
- les citations représentatives ;
- les limites de qualité et les points à auditer.

## 11. Validation Humaine

La carte ne doit pas être considérée valide parce qu'elle est lisible. Il faut un protocole de validation explicite.

### Protocole minimal

1. Échantillonner des commentaires par source, période, volume, communauté et outliers.
2. Faire annoter humainement thèmes, frames, cibles, stances, claims et ambiguïtés.
3. Comparer extraction LLM et annotation humaine.
4. Mesurer l'accord inter-annotateurs sur un sous-échantillon doublement annoté.
5. Auditer qualitativement les erreurs les plus fréquentes.
6. Mesurer le recall des discours minoritaires.
7. Tester la stabilité entre prompts, modèles et seeds.
8. Comparer avec les baselines embedding, BERTopic et TF-IDF.

### Métriques

- **Macro-F1** sur frames, stances et targets.
- **Cohen's kappa** ou **Krippendorff's alpha** pour l'accord humain.
- **Cluster purity** si des labels humains de communauté existent.
- **Normalized mutual information** entre communautés automatiques et labels humains, si disponible.
- **Stabilité Jaccard** des communautés entre prompts, modèles ou périodes.
- **Coverage des discours minoritaires** : part des catégories rares correctement détectées.
- **Proportion de bruit/outliers** : nécessaire pour éviter les clusters forcés.
- **Audit qualitatif des citations** : les citations justifient-elles vraiment les labels ?

### Points d'audit prioritaires

- stances mal conditionnées par la cible ;
- confusion entre source_actor et cible ;
- surinterprétation de sarcasme ;
- claims canoniques trop généraux ;
- frames qui naturalisent une catégorie politique discutable ;
- communautés dominées par un nom propre plutôt que par une configuration discursive.

## 12. Baselines

La V2.6.3 doit être évaluée contre des baselines simples. Sinon, on ne saura pas si le graphe apporte vraiment quelque chose.

| Baseline | Description | Ce qu'elle teste |
| --- | --- | --- |
| Embedding brut + KMeans | Clustering direct des commentaires. | Baseline faible mais lisible. |
| Embedding brut + HDBSCAN | Clustering avec bruit autorisé. | Capacité à ne pas forcer les petits volumes. |
| BERTopic | Topics par embeddings + c-TF-IDF. | Qualité topic modeling standard. |
| LLM direct en labels | Le LLM nomme directement les groupes. | Risque de labels séduisants mais peu structurés. |
| LLM reformulation + embedding | Embeddings sur reformulations canoniques. | Effet de la normalisation textuelle. |
| Matrice acteur × cadrage × temps | Agrégation simple sans communauté. | Valeur des tableaux interprétables. |
| Hypergraphe sans message passing | Graphe pondéré + communautés. | Baseline V2.6.3 recommandée. |
| Hypergraphe avec message passing | GNN ou propagation relationnelle. | Gain éventuel d'une approche neuronale. |

Critères de comparaison :

- stabilité ;
- interprétabilité ;
- capacité à retrouver des discours minoritaires ;
- qualité des citations représentatives ;
- proportion d'outliers ;
- coût et latence ;
- facilité d'audit.

## 13. Garde-Fous Méthodologiques, Politiques Et Juridiques

Cette infrastructure doit rester une analyse agrégée de circulation discursive.

### À affirmer explicitement

- Les commentaires publics ne sont pas représentatifs de l'électorat.
- L'analyse porte sur la circulation de discours publics, pas sur l'opinion générale.
- Les résultats décrivent des configurations observées dans un corpus donné.
- Une proximité de graphe n'est pas une preuve causale.
- Une communauté discursive n'est pas un groupe social réel.
- Un commentaire ne doit pas servir à profiler un individu.

### À interdire dans le produit

- Scorer des personnes ou des auteurs de commentaires.
- Inférer des attributs sensibles individuels.
- Prédire le vote ou la persuasion individuelle.
- Faire du microciblage politique.
- Présenter les communautés comme des segments psychographiques.
- Utiliser les citations pour exposer inutilement des individus.

### Pratiques de minimisation

- Pseudonymiser ou supprimer les identifiants auteur.
- Conserver les citations uniquement quand elles servent l'audit.
- Limiter la longueur des citations exportées.
- Agréger les métriques à un niveau suffisant.
- Documenter les biais de collecte, de plateforme et de période.
- Séparer diagnostic stratégique, exploration scientifique et preuve causale.

## 14. Architecture Technique

### Modules Python proposés

```text
observatoire/
  discourse_extraction.py      # extraction LLM structurée en JSON
  discourse_schema.py          # schémas, validation, versions
  discourse_canonicalization.py# normalisation frames/claims/targets
  discourse_graph.py           # construction bipartite/hypergraphe
  discourse_weights.py         # IDF, relation weights, pénalités
  discourse_similarity.py      # matrices d'incidence et K hybride
  discourse_communities.py     # Leiden/Louvain/spectral/HDBSCAN
  discourse_temporal.py        # graphes G_t et métriques dynamiques
  discourse_reports.py         # exports CSV/Markdown/HTML
```

### Flux de données

1. Ingestion de commentaires publics et métadonnées.
2. Nettoyage, déduplication, anonymisation ou pseudonymisation.
3. Filtrage des verbatims trop pauvres pour l'analyse discursive.
4. Extraction LLM structurée vers JSONL.
5. Validation du schéma et rejet des sorties invalides.
6. Canonicalisation des claims, frames, cibles et familles argumentatives.
7. Stockage JSONL et Parquet.
8. Construction du graphe biparti ou de l'hypergraphe.
9. Calcul IDF, poids relationnels et pénalités de nœuds vagues.
10. Construction des matrices d'incidence sparse.
11. Calcul de `K` et détection de communautés.
12. Exports analytiques, visualisations et rapport qualité.

### Librairies

- **Stockage** : JSONL pour audit, Parquet pour traitements analytiques.
- **Validation** : pydantic ou dataclasses + validation explicite.
- **Graphes** : NetworkX pour POC, igraph/leidenalg pour performance.
- **Hypergraphes** : HyperNetX en expérimentation.
- **Matrices** : scipy sparse.
- **Embeddings** : sentence-transformers pour local, API embeddings si nécessaire.
- **Clustering** : Leiden/Louvain, clustering spectral, HDBSCAN.
- **Visualisation** : Plotly, PyVis, export Gephi.
- **Dashboard** : Streamlit ou HTML statique.

### Outputs proposés

```text
outputs/discursive_units.jsonl
outputs/discursive_units.parquet
outputs/discursive_nodes.csv
outputs/discursive_edges.csv
outputs/discursive_incidence_frame.npz
outputs/discursive_incidence_claim.npz
outputs/discursive_incidence_stance.npz
outputs/discursive_incidence_target.npz
outputs/discursive_incidence_argument.npz
outputs/discursive_incidence_tone.npz
outputs/discursive_similarity_edges.csv
outputs/discursive_communities.csv
outputs/discursive_community_profiles.md
outputs/discursive_temporal_metrics.csv
outputs/discursive_validation_sample.csv
```

Le POC V2.6.3 implémente déjà la partie statique : unités, nœuds, arêtes, matrices d'incidence, similarité hybride, communautés et profils Markdown. Les sorties temporelles et l'échantillon de validation restent des extensions à ajouter.

## 15. POC En 3 À 4 Semaines

### Semaine 1 — Extraction et schéma

- Collecter un corpus public limité mais diversifié.
- Nettoyer, dédupliquer et anonymiser.
- Stabiliser le prompt d'extraction LLM.
- Produire les premiers JSON validés.
- Auditer manuellement un petit échantillon.
- Corriger le schéma si les distinctions source/cible/stance restent floues.

### Semaine 2 — Graphe et pondération

- Canonicaliser frames, claims et targets.
- Construire le graphe biparti typé.
- Implémenter IDF, relation weights et pénalisation des nœuds vagues.
- Produire les premières matrices d'incidence.
- Comparer avec clustering embeddings et BERTopic.
- Identifier les nœuds hubs problématiques.

### Semaine 3 — Communautés et visualisation

- Calculer `K` hybride.
- Tester Leiden/Louvain, spectral et HDBSCAN.
- Produire les profils de communautés : frames, claims, targets, stances, citations.
- Construire les matrices `source_actor × frame` et `target × stance`.
- Ajouter une visualisation Plotly/PyVis ou export Gephi.
- Réaliser un audit humain ciblé des communautés.

### Semaine 4 — Robustesse et rapport

- Tester stabilité entre prompts, modèles et seeds.
- Mesurer coverage, outliers et stabilité Jaccard.
- Produire un rapport final avec limites.
- Documenter les garde-fous.
- Décider si le message passing est justifié ou non.

## 16. Critique Théorique

Un discours ne se réduit pas à un cluster. Un cluster est une opération géométrique ; un discours est une configuration sociale, rhétorique et contextuelle. La carte ne doit donc pas être prise pour le territoire.

L'hypergraphe est plus fidèle que le clustering direct parce qu'il préserve la structure relationnelle : qui parle sous quelle source, de quoi, avec quel cadrage, contre quelle cible, avec quelle stance et quel type d'argument. Mais cette fidélité dépend de l'ontologie choisie. Si les catégories sont mauvaises, le graphe sera propre, stable et trompeur.

Risques principaux :

- **Naturalisation du codebook** : les catégories du schéma peuvent apparaître comme naturelles alors qu'elles sont construites.
- **Propagation d'erreurs** : une erreur d'extraction LLM devient une arête, puis influence des communautés.
- **Surinterprétation visuelle** : une carte lisible peut donner une impression de preuve.
- **Réduction de l'ambiguïté** : les commentaires ironiques, ambivalents ou pauvres peuvent être artificiellement clarifiés.
- **Confusion exploration/preuve** : une communauté détectée est une hypothèse d'analyse, pas une démonstration.

La bonne posture est donc de séparer trois niveaux :

1. **Exploration** : détecter des motifs, formuler des hypothèses.
2. **Interprétation** : lire les citations, nommer prudemment les configurations.
3. **Preuve** : valider avec annotation humaine, stabilité et baselines.

## 17. Recommandation Finale

La V2.6.3 doit partir sur un hypergraphe discursif interprétable, implémenté pragmatiquement comme graphe biparti typé. C'est le meilleur compromis entre expressivité, auditabilité et vitesse de prototypage.

Ordre recommandé :

1. Extraction LLM structurée avec citations obligatoires.
2. Schéma JSON versionné et conservateur.
3. Graphe biparti typé commentaire-attribut.
4. Pondération IDF, poids relationnels et pénalisation des hubs vagues.
5. Similarité hybride `K` entre commentaires.
6. Détection de communautés non-neuronale.
7. Profils de communautés avec citations et scores de qualité.
8. Validation humaine et comparaison aux baselines.

Le message passing neuronal ne doit pas être le cœur de la V2.6.3. Il pourra être testé plus tard si trois conditions sont réunies : volume suffisant, labels de validation fiables et tâche supervisée précise. Sans cela, il risque surtout de rendre le système moins explicable, plus fragile et plus difficile à contester.

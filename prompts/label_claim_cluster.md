Tu nommes une famille de claims extraite inductivement de commentaires YouTube politiques.

Règles :
- Donne un label court, interprétable et non sensationnaliste.
- Décris le point commun des claims, pas l'opinion générale.
- Mentionne les variations internes si elles existent.
- Signale le bruit ou les contre-exemples si les claims sont hétérogènes.
- Ne transforme pas les résultats en preuve causale ou en mesure représentative.

Format JSON obligatoire :
{
  "cluster_label": "label court",
  "summary": "résumé prudent en une ou deux phrases",
  "internal_variations": ["variation 1", "variation 2"],
  "representative_claims": ["claim 1", "claim 2"],
  "counterexamples_or_noise": ["élément bruité éventuel"]
}


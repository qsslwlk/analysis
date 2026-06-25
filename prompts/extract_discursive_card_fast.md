Tu produis une fiche discursive courte depuis un commentaire YouTube politique.

Objectif :
- extraire seulement les informations utiles au graphe discursif ;
- rester strictement ancré dans le texte fourni ;
- répondre vite en JSON valide.

Règles :
- Si le commentaire ne contient pas de contenu discursif exploitable, retourne `{"discursive_card": null}`.
- N'invente pas de thème, cible ou argument absent du commentaire.
- Les citations représentatives sont obligatoires dès qu'une fiche est produite.
- Privilégie des labels courts et stables.
- Réponds uniquement en JSON valide.

Format JSON obligatoire :
{
  "discursive_card": {
    "theme_main": "thème principal court",
    "subthemes": [],
    "dominant_frame": "cadrage dominant court",
    "secondary_frames": [],
    "stance_targets": [
      {
        "target": "cible identifiable ou vide",
        "stance": "adhésion | rejet | scepticisme | ironie | demande | autre",
        "evidence": "citation courte",
        "confidence": 0.0
      }
    ],
    "central_argument": "argument central court ou vide",
    "argument_type": "économique | moral | institutionnel | identitaire | programmatique | stratégique | témoignage | autre | aucun",
    "attack_or_objection": "",
    "emotion_tone": "tonalité courte",
    "ambiguities": [],
    "representative_quotes": ["citation courte"],
    "confidence": 0.0,
    "discursive_summary": "résumé sobre en une phrase courte"
  }
}

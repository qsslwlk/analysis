import json
import unittest

import pandas as pd

from observatoire.discourse_taxonomy import normalize_discursive_cards


class DiscourseTaxonomyTest(unittest.TestCase):
    def test_normalize_discursive_cards_adds_controlled_fields(self):
        cards = pd.DataFrame(
            [
                {
                    "comment_id": "c1",
                    "dominant_frame": "justice sociale et fiscale",
                    "argument_type": "économique",
                    "emotion_tone": "inquiétude, indignation",
                    "attack_or_objection": "objection",
                    "stance_targets_json": json.dumps(
                        [
                            {
                                "target": "Le Gouvernement",
                                "stance": "rejet",
                                "evidence": "les charges augmentent",
                            }
                        ],
                        ensure_ascii=False,
                    ),
                    "representative_quotes_json": json.dumps(["les charges augmentent"], ensure_ascii=False),
                    "ambiguities_json": "[]",
                }
            ]
        )

        normalized = normalize_discursive_cards(cards)
        stances = json.loads(normalized.iloc[0]["stance_targets_controlled_json"])
        targets = json.loads(normalized.iloc[0]["targets_normalized_json"])

        self.assertEqual(normalized.iloc[0]["macro_frame"], "justice_sociale")
        self.assertEqual(normalized.iloc[0]["frame_primary"], "justice_fiscale")
        self.assertEqual(normalized.iloc[0]["argument_family_controlled"], "economique")
        self.assertEqual(normalized.iloc[0]["tone_controlled"], "concern")
        self.assertEqual(normalized.iloc[0]["rhetorical_register"], "question")
        self.assertEqual(stances[0]["stance"], "hostile")
        self.assertEqual(targets, ["le gouvernement"])
        self.assertGreater(normalized.iloc[0]["evidence_score"], 0)
        self.assertGreater(normalized.iloc[0]["taxonomy_fit_score"], 0.8)


if __name__ == "__main__":
    unittest.main()

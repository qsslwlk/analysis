import unittest

from observatoire.text_quality import is_semantic_candidate, semantic_exclusion_reason, text_quality_metrics


class TextQualityTest(unittest.TestCase):
    def test_emoji_only_is_excluded(self):
        self.assertEqual(semantic_exclusion_reason("👍🏻👍🏻👍🏻"), "no_alpha_tokens")

    def test_short_reaction_is_excluded(self):
        self.assertEqual(semantic_exclusion_reason("Bravo JLM merci"), "reaction_or_name_only")

    def test_short_comment_is_excluded(self):
        self.assertEqual(semantic_exclusion_reason("Ah la démagogie. Très pure!"), "too_short")

    def test_argumentative_comment_is_candidate(self):
        text = (
            "C'est bien beau tout ça mais le financement vous le trouvez comment ? "
            "Il faudrait expliquer les recettes et les hypothèses budgétaires."
        )
        self.assertTrue(is_semantic_candidate(text))

    def test_metrics_count_meaningful_tokens(self):
        metrics = text_quality_metrics("Baisse de la TVA sur les carburants et financement du programme.")
        self.assertGreaterEqual(metrics["meaningful_token_count"], 5)


if __name__ == "__main__":
    unittest.main()

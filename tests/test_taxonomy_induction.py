import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def _load_taxonomy_induction_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "taxonomy_induction.py"
    spec = importlib.util.spec_from_file_location("taxonomy_induction", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


taxonomy_induction = _load_taxonomy_induction_module()


class TaxonomyInductionTest(unittest.TestCase):
    def test_heuristic_run_without_nodes_preserves_source_taxonomy_and_outputs_valid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taxonomy_path = tmp_path / "taxonomy.json"
            source_taxonomy = Path("config/discourse_taxonomy.example.json").read_text(encoding="utf-8")
            taxonomy_path.write_text(source_taxonomy, encoding="utf-8")
            before_hash = hashlib.sha256(taxonomy_path.read_bytes()).hexdigest()

            units = pd.DataFrame(
                [
                    {
                        "comment_id": "c1",
                        "actor": "RN",
                        "dominant_frame": "Justice sociale et fiscale",
                        "dominant_frame_free": "Justice sociale et fiscale",
                        "macro_frame": "other",
                        "frame_primary": "other",
                        "argument_type": "persuasion",
                        "argument_family_controlled": "other",
                        "emotion_tone": "optimiste",
                        "tone_controlled": "other",
                        "attack_or_objection": "objection",
                        "rhetorical_register": "other",
                        "raw_comment_preview": "Il faut convaincre pour obtenir une vraie justice sociale et fiscale.",
                    },
                    {
                        "comment_id": "c2",
                        "actor": "LFI",
                        "dominant_frame": "other",
                        "dominant_frame_free": "other",
                        "macro_frame": "other",
                        "frame_primary": "other",
                        "argument_type": "other",
                        "argument_family_controlled": "other",
                        "emotion_tone": "other",
                        "tone_controlled": "other",
                        "attack_or_objection": "other",
                        "rhetorical_register": "other",
                        "raw_comment_preview": "Bravo.",
                    },
                    {
                        "comment_id": "c3",
                        "actor": "RN",
                        "dominant_frame": "journaliste",
                        "dominant_frame_free": "journaliste",
                        "macro_frame": "other",
                        "frame_primary": "other",
                        "argument_type": "other",
                        "argument_family_controlled": "other",
                        "emotion_tone": "other",
                        "tone_controlled": "other",
                        "attack_or_objection": "other",
                        "rhetorical_register": "other",
                        "raw_comment_preview": "La journaliste coupe la parole.",
                    },
                ]
            )
            units_path = tmp_path / "discursive_units.csv"
            units.to_csv(units_path, index=False)
            output_dir = tmp_path / "taxonomy_induction"

            paths = taxonomy_induction.run_induction(
                argparse.Namespace(
                    taxonomy=taxonomy_path,
                    units=units_path,
                    nodes=None,
                    community_summary=None,
                    attribute_contributions=None,
                    output_dir=output_dir,
                    mode="heuristic",
                    llm_provider="none",
                    llm_model="none",
                    llm_api_key=None,
                    llm_base_url=None,
                    min_count=1,
                    apply_min_confidence=0.75,
                )
            )

            after_hash = hashlib.sha256(taxonomy_path.read_bytes()).hexdigest()
            self.assertEqual(before_hash, after_hash)
            for path in paths.values():
                self.assertTrue(path.exists())

            candidates = json.loads(paths["candidates"].read_text(encoding="utf-8"))
            self.assertIn("candidate_aliases", candidates)
            self.assertIn("other_rates", candidates)

            remap = pd.read_csv(paths["remap_table"])
            expected_columns = set(taxonomy_induction.REMAP_COLUMNS)
            self.assertTrue(expected_columns.issubset(remap.columns))
            self.assertIn("keep_other", set(remap["action"]))
            self.assertFalse(
                remap[
                    (remap["raw_label"].fillna("").str.lower() == "other")
                    & (remap["action"] != "keep_other")
                ].any(axis=None)
            )

            remapped = pd.read_csv(paths["remapped_units"])
            self.assertIn("frame_primary_raw", remapped.columns)
            self.assertIn("frame_primary_remapped", remapped.columns)
            self.assertEqual(remapped.loc[remapped["comment_id"] == "c1", "frame_primary_remapped"].iloc[0], "justice_fiscale")
            self.assertEqual(remapped.loc[remapped["comment_id"] == "c2", "frame_primary_remapped"].iloc[0], "other")
            self.assertEqual(remapped.loc[remapped["comment_id"] == "c3", "frame_primary_remapped"].iloc[0], "other")

            report = paths["report"].read_text(encoding="utf-8")
            self.assertIn("Taux de `other` avant/après", report)
            self.assertIn("Commentaires affectés", report)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import math
import unittest
from pathlib import Path

import pandas as pd


def _load_bridge_report_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_bridge_report.py"
    spec = importlib.util.spec_from_file_location("generate_bridge_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bridge_report = _load_bridge_report_module()


class BridgeReportTest(unittest.TestCase):
    def test_balance_and_specificity_are_positive_and_interpretable(self):
        self.assertEqual(bridge_report.compute_balance(10, 10), 1.0)
        self.assertAlmostEqual(bridge_report.compute_balance(10, 0), 0.0)

        specific = bridge_report.compute_specificity(total_units=100, df_comments=4)
        generic = bridge_report.compute_specificity(total_units=100, df_comments=80)
        self.assertGreater(specific, generic)
        self.assertGreater(specific, 0)

    def test_make_report_rows_adds_quotes_and_audit_fields(self):
        bridges = pd.DataFrame(
            [
                {
                    "attribute_type": "FrameNode",
                    "attribute_label": "justice_fiscale",
                    "df_comments": 4,
                    "n_RN_comments": 2,
                    "n_LFI_comments": 2,
                    "bridge_score": 12.0,
                    "top_RN_comment_ids": '["rn1"]',
                    "top_LFI_comment_ids": '["lfi1"]',
                    "top_RN_summaries": "[]",
                    "top_LFI_summaries": "[]",
                }
            ]
        )
        units = pd.DataFrame(
            [
                {
                    "comment_id": "rn1",
                    "actor": "RN",
                    "discursive_summary": "Résumé RN",
                    "raw_comment_preview": "Citation RN",
                    "frame_primary": "justice_fiscale",
                },
                {
                    "comment_id": "lfi1",
                    "actor": "LFI",
                    "discursive_summary": "Résumé LFI",
                    "raw_comment_preview": "Citation LFI",
                    "frame_primary": "justice_fiscale",
                },
            ]
        )

        report = bridge_report.make_report_rows(bridges, units, "RN", "LFI", max_quotes=2)

        self.assertEqual(len(report), 1)
        row = report.iloc[0]
        self.assertEqual(row["bridge_label"], "justice_fiscale")
        self.assertEqual(row["bridge_category"], "content")
        self.assertEqual(row["audit_label"], "")
        self.assertIn("Citation RN", row["top_RN_quotes_short"])
        self.assertIn("Citation LFI", row["top_LFI_quotes_short"])
        self.assertTrue(math.isfinite(float(row["report_score"])))


if __name__ == "__main__":
    unittest.main()

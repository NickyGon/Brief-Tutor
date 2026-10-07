import tempfile
import unittest
from pathlib import Path

from graph.brief_naming import parse_brief_filename
from graph.tools import (
    classify_campaign_path_hierarchy,
    extract_campaign_instance_id,
    extract_family_slug_from_filename,
    list_same_family_local_spreadsheets,
)
from graph.supabase.repository import SupabaseWorkflowRepository


class BriefFilenameParsingTests(unittest.TestCase):
    def test_parse_filename_supports_a_and_d_tokens(self) -> None:
        a_file = "2025-11-rogerbeasleyvolvovcna-A-25008537.xlsx"
        d_file = "2026-06-tonydivinousedcarsntrucks-D-94095.xlsx"

        a_parsed = parse_brief_filename(a_file)
        d_parsed = parse_brief_filename(d_file)

        self.assertIsNotNone(a_parsed)
        self.assertIsNotNone(d_parsed)
        self.assertEqual(a_parsed["account_id"], "rogerbeasleyvolvovcna")
        self.assertEqual(a_parsed["campaign_token"], "A-25008537")
        self.assertEqual(d_parsed["account_id"], "tonydivinousedcarsntrucks")
        self.assertEqual(d_parsed["campaign_token"], "D-94095")

    def test_existing_helpers_use_shared_parser(self) -> None:
        file_name = "2026-06-tonydivinousedcarsntrucks-D-94095.xlsx"
        self.assertEqual(extract_family_slug_from_filename(file_name), "tonydivinousedcarsntrucks")
        self.assertEqual(extract_campaign_instance_id(file_name), "D-94095")


class HierarchicalSearchTests(unittest.TestCase):
    def test_grouped_account_hierarchy_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = (
                root
                / "Campaigns"
                / "CentralTexasGroup"
                / "rogerbeasleyvolvovcna"
                / "2025-11-rogerbeasleyvolvovcna-A-25008537.xlsx"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

            context = classify_campaign_path_hierarchy(str(target))
            self.assertTrue(context["has_group_folder"])
            self.assertIn("Campaigns", context["campaigns_root"])
            self.assertEqual(context["account_id"], "rogerbeasleyvolvovcna")

    def test_staged_candidate_scope_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaigns = root / "Campaigns"
            group = campaigns / "CentralTexasGroup"
            account = group / "rogerbeasleyvolvovcna"
            sibling_account = group / "tonydivinousedcarsntrucks"
            other_group_account = campaigns / "OtherGroup" / "anotheraccount"

            target = account / "2025-11-rogerbeasleyvolvovcna-A-25008537.xlsx"
            same_account_other = account / "2025-12-rogerbeasleyvolvovcna-D-25008538.xlsx"
            group_sibling = sibling_account / "2026-06-tonydivinousedcarsntrucks-D-94095.xlsx"
            campaigns_sibling = other_group_account / "2026-07-anotheraccount-A-11111111.xlsx"

            for file_path in [target, same_account_other, group_sibling, campaigns_sibling]:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()

            account_scope = list_same_family_local_spreadsheets(str(target), search_scope="account")
            group_scope = list_same_family_local_spreadsheets(str(target), search_scope="group")
            campaigns_scope = list_same_family_local_spreadsheets(str(target), search_scope="campaigns")

            self.assertIn(str(same_account_other.resolve()), account_scope)
            self.assertNotIn(str(group_sibling.resolve()), account_scope)

            self.assertIn(str(group_sibling.resolve()), group_scope)
            self.assertNotIn(str(same_account_other.resolve()), group_scope)

            self.assertIn(str(campaigns_sibling.resolve()), campaigns_scope)
            self.assertNotIn(str(same_account_other.resolve()), campaigns_scope)


class OemCompatibilityTests(unittest.TestCase):
    def test_oem_compatibility_rules(self) -> None:
        source = {"handles_all_oems": False, "oems": ["Ford"], "oem_family": "Ford-Lincoln"}
        same_oem = {"handles_all_oems": False, "oems": ["ford"], "oem_family": "NA"}
        same_family = {"handles_all_oems": False, "oems": [], "oem_family": "Ford-Lincoln"}
        all_oems = {"handles_all_oems": True, "oems": [], "oem_family": "NA"}
        mismatch = {"handles_all_oems": False, "oems": ["GM"], "oem_family": "GM Motors"}

        self.assertTrue(SupabaseWorkflowRepository.oem_compatible(source, same_oem))
        self.assertTrue(SupabaseWorkflowRepository.oem_compatible(source, same_family))
        self.assertTrue(SupabaseWorkflowRepository.oem_compatible(source, all_oems))
        self.assertFalse(SupabaseWorkflowRepository.oem_compatible(source, mismatch))


if __name__ == "__main__":
    unittest.main()

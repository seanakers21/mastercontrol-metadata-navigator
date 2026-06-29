import unittest

from data_service import load_events
from search_service import apply_event_filters, lexical_score, rank_events


class SearchServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = load_events()

    def test_dry_pump_question_finds_known_event(self):
        results = rank_events(
            "Have centrifugal pumps run dry and broken in the past?",
            self.events,
        )
        ids = [item["record_id"] for item, _ in results[:4]]
        self.assertIn("DEV-2023-0142", ids)
        self.assertIn("VAR-2024-0087", ids)

    def test_exact_asset_id_is_searchable(self):
        results = rank_events("What happened to P-204?", self.events)
        self.assertEqual(results[0][0]["record_id"], "DEV-2023-0142")

    def test_filters_are_applied_before_ranking(self):
        filtered = apply_event_filters(
            self.events,
            {
                "site": ["South Campus"],
                "event_type": [],
                "department": [],
                "status": [],
                "year_range": (2025, 2025),
            },
        )
        self.assertTrue(filtered)
        self.assertTrue(all(item["site"] == "South Campus" for item in filtered))
        self.assertTrue(all(item["event_date"].startswith("2025") for item in filtered))

    def test_empty_query_has_no_score(self):
        self.assertEqual(lexical_score("", "centrifugal pump"), 0.0)


if __name__ == "__main__":
    unittest.main()

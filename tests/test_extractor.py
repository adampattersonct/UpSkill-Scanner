"""Tests for UpskillScan extractor and related modules."""

import json

import pandas as pd
import pytest

from extractor import SkillExtraction
from cost_tracker import CostTracker, CostLimitExceeded
from aggregator import SkillAggregator


# --- SkillExtraction Pydantic model ---


class TestSkillExtraction:
    def test_parse_valid_json(self):
        raw = json.dumps(
            {
                "technical_skills": ["Python", "SQL"],
                "soft_skills": ["Communication"],
                "certifications": ["CFA"],
                "domain_knowledge": ["Econometrics"],
            }
        )
        result = SkillExtraction.model_validate_json(raw)
        assert result.technical_skills == ["Python", "SQL"]
        assert result.soft_skills == ["Communication"]
        assert result.certifications == ["CFA"]
        assert result.domain_knowledge == ["Econometrics"]

    def test_parse_empty_lists(self):
        raw = json.dumps(
            {
                "technical_skills": [],
                "soft_skills": [],
                "certifications": [],
                "domain_knowledge": [],
            }
        )
        result = SkillExtraction.model_validate_json(raw)
        assert result.technical_skills == []

    def test_parse_missing_fields_uses_defaults(self):
        raw = json.dumps({"technical_skills": ["Excel"]})
        result = SkillExtraction.model_validate_json(raw)
        assert result.technical_skills == ["Excel"]
        assert result.soft_skills == []
        assert result.certifications == []
        assert result.domain_knowledge == []

    def test_parse_invalid_json_raises(self):
        with pytest.raises(Exception):
            SkillExtraction.model_validate_json("not json")


# --- CostTracker ---


class TestCostTracker:
    def test_initial_cost_is_zero(self):
        tracker = CostTracker(model="claude-haiku-4-5-20251001", cost_limit=5.0)
        assert tracker.estimated_cost == 0.0

    def test_update_from_usage(self):
        tracker = CostTracker(model="claude-haiku-4-5-20251001", cost_limit=5.0)
        # 1M input tokens at $0.50/MTok = $0.50
        tracker.update_from_usage(1_000_000, 0)
        assert abs(tracker.estimated_cost - 0.50) < 0.01

    def test_cost_limit_exceeded(self):
        tracker = CostTracker(model="claude-haiku-4-5-20251001", cost_limit=0.01)
        tracker.update_from_usage(1_000_000, 1_000_000)  # Well over $0.01
        with pytest.raises(CostLimitExceeded):
            tracker.check_limit()

    def test_estimate_batch_cost(self):
        tracker = CostTracker(model="claude-haiku-4-5-20251001", cost_limit=5.0)
        est = tracker.estimate_batch_cost(
            num_requests=1000, avg_input_tokens=1500, avg_output_tokens=300
        )
        # 1000 * 1500 = 1.5M input tokens * $0.50/MTok = $0.75
        # 1000 * 300 = 0.3M output tokens * $2.50/MTok = $0.75
        # Total = $1.50
        assert abs(est - 1.50) < 0.01

    def test_sync_pricing_is_double(self):
        tracker = CostTracker(
            model="claude-haiku-4-5-20251001", cost_limit=5.0, use_batch=False
        )
        tracker.update_from_usage(1_000_000, 0)
        # Sync: $1.00/MTok input
        assert abs(tracker.estimated_cost - 1.00) < 0.01

    def test_budget_remaining(self):
        tracker = CostTracker(model="claude-haiku-4-5-20251001", cost_limit=5.0)
        tracker.update_from_usage(1_000_000, 0)
        assert abs(tracker.budget_remaining - 4.50) < 0.01


# --- SkillAggregator ---


class TestSkillAggregator:
    def _make_df(self):
        return pd.DataFrame(
            {
                "title": ["Job A", "Job B", "Job C"],
                "region": ["Northeast", "Northeast", "West"],
                "technical_skills": [
                    json.dumps(["Python", "SQL"]),
                    json.dumps(["Python", "Excel"]),
                    json.dumps(["SQL", "R"]),
                ],
                "soft_skills": [
                    json.dumps(["Communication"]),
                    json.dumps(["Communication", "Teamwork"]),
                    json.dumps(["Leadership"]),
                ],
                "certifications": [
                    json.dumps([]),
                    json.dumps(["CFA"]),
                    json.dumps([]),
                ],
                "domain_knowledge": [
                    json.dumps(["Econometrics"]),
                    json.dumps([]),
                    json.dumps(["Data Analysis"]),
                ],
            }
        )

    def test_aggregate_national(self):
        agg = SkillAggregator()
        df = self._make_df()
        result = agg.aggregate_national(df)
        assert len(result) > 0
        # Python appears in 2 out of 3 postings
        python_row = result[result["skill"] == "Python"]
        assert len(python_row) == 1
        assert python_row.iloc[0]["count"] == 2

    def test_aggregate_by_region(self):
        agg = SkillAggregator()
        df = self._make_df()
        result = agg.aggregate_by_region(df)
        assert "Northeast" in result.columns
        assert "West" in result.columns

    def test_aggregate_by_category(self):
        agg = SkillAggregator()
        df = self._make_df()
        result = agg.aggregate_by_category(df)
        assert len(result) == 4  # 4 categories
        tech_row = result[result["category"] == "Technical"]
        assert tech_row.iloc[0]["unique_skills"] == 4  # Python, SQL, Excel, R

    def test_compute_statistics(self):
        agg = SkillAggregator()
        df = self._make_df()
        stats = agg.compute_statistics(df)
        assert stats["total_postings"] == 3
        assert stats["postings_with_skills"] == 3
        assert stats["unique_skills"] > 0

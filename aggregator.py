"""Counting, ranking, and regional slicing for UpskillScan."""

import json
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd


SKILL_CATEGORIES = ["technical_skills", "soft_skills", "certifications", "domain_knowledge"]

CATEGORY_LABELS = {
    "technical_skills": "Technical",
    "soft_skills": "Soft Skills",
    "certifications": "Certifications",
    "domain_knowledge": "Domain Knowledge",
}


class SkillAggregator:
    """Aggregates extracted skills into rankings and matrices."""

    def _parse_skills(self, val) -> List[str]:
        """Parse a JSON-encoded skill list from a DataFrame cell."""
        if pd.isna(val):
            return []
        try:
            skills = json.loads(val) if isinstance(val, str) else val
            return skills if isinstance(skills, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def _count_skills(
        self, df: pd.DataFrame, categories: List[str] = None
    ) -> List[Tuple[str, str, int]]:
        """Count skill occurrences. Returns list of (skill, category, count)."""
        categories = categories or SKILL_CATEGORIES
        results = []
        for cat in categories:
            if cat not in df.columns:
                continue
            counter = Counter()
            for val in df[cat]:
                for skill in self._parse_skills(val):
                    counter[skill] += 1
            for skill, count in counter.items():
                results.append((skill, cat, count))
        return results

    def aggregate_national(self, df: pd.DataFrame) -> pd.DataFrame:
        """Overall top skills across all postings."""
        counts = self._count_skills(df)
        if not counts:
            return pd.DataFrame(columns=["skill", "category", "count", "percentage"])

        result = pd.DataFrame(counts, columns=["skill", "category", "count"])
        # Combine counts for same skill across categories (shouldn't happen, but just in case)
        # Keep category from highest count
        result = result.sort_values("count", ascending=False)
        result["percentage"] = (result["count"] / len(df) * 100).round(1)
        result["category_label"] = result["category"].map(CATEGORY_LABELS)
        return result.reset_index(drop=True)

    def aggregate_by_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """Skills x Region matrix with percentages."""
        if "region" not in df.columns:
            print("  Warning: No 'region' column found")
            return pd.DataFrame()

        regions = df["region"].unique()
        all_rows = []

        for region in sorted(regions):
            region_df = df[df["region"] == region]
            region_total = len(region_df)
            counts = self._count_skills(region_df)
            for skill, cat, count in counts:
                pct = round(count / region_total * 100, 1) if region_total > 0 else 0
                all_rows.append(
                    {
                        "skill": skill,
                        "category": cat,
                        "region": region,
                        "count": count,
                        "percentage": pct,
                    }
                )

        if not all_rows:
            return pd.DataFrame()

        result = pd.DataFrame(all_rows)
        # Pivot to matrix: skill rows, region columns
        matrix = result.pivot_table(
            index="skill",
            columns="region",
            values="percentage",
            aggfunc="sum",
            fill_value=0,
        )
        # Add total count for sorting
        skill_totals = result.groupby("skill")["count"].sum()
        matrix["total_count"] = skill_totals
        matrix = matrix.sort_values("total_count", ascending=False)
        return matrix

    def aggregate_by_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Breakdown by skill category."""
        rows = []
        for cat in SKILL_CATEGORIES:
            if cat not in df.columns:
                continue
            all_skills = set()
            total_mentions = 0
            for val in df[cat]:
                skills = self._parse_skills(val)
                all_skills.update(skills)
                total_mentions += len(skills)
            rows.append(
                {
                    "category": CATEGORY_LABELS.get(cat, cat),
                    "unique_skills": len(all_skills),
                    "total_mentions": total_mentions,
                    "avg_per_posting": round(total_mentions / len(df), 1) if len(df) > 0 else 0,
                }
            )
        return pd.DataFrame(rows)

    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Summary statistics for the research paper."""
        total_postings = len(df)

        # Count unique skills across all categories
        all_skills = set()
        total_skill_mentions = 0
        for cat in SKILL_CATEGORIES:
            if cat not in df.columns:
                continue
            for val in df[cat]:
                skills = self._parse_skills(val)
                all_skills.update(skills)
                total_skill_mentions += len(skills)

        # Postings with at least one extracted skill
        has_skills = 0
        for _, row in df.iterrows():
            for cat in SKILL_CATEGORIES:
                if cat in df.columns and self._parse_skills(row.get(cat)):
                    has_skills += 1
                    break

        # Regional breakdown
        region_counts = {}
        if "region" in df.columns:
            region_counts = df["region"].value_counts().to_dict()

        stats = {
            "total_postings": total_postings,
            "postings_with_skills": has_skills,
            "extraction_rate": round(has_skills / total_postings * 100, 1) if total_postings > 0 else 0,
            "unique_skills": len(all_skills),
            "total_skill_mentions": total_skill_mentions,
            "avg_skills_per_posting": round(total_skill_mentions / total_postings, 1) if total_postings > 0 else 0,
            "postings_by_region": region_counts,
        }
        return stats

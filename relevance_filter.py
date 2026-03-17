"""
Relevance Filter — Remove off-topic job postings before LLM extraction.

JobSpy returns loosely-matched results (e.g., searching "Economic Associate"
may return Walmart store leads, nurses, cooks). This module filters out
postings whose titles are clearly unrelated to the search intent using
a block-list of known off-topic patterns.

The block list is organized into categories. Categories that overlap with the
user's search titles are automatically disabled, making the filter
discipline-agnostic. For example, searching for "Registered Nurse" disables
the healthcare category so nursing postings are preserved.

Can be used in the pipeline (main.py) or standalone:
    python relevance_filter.py --input output/raw_jobs.csv --titles "Economist,Policy Analyst"
"""

import argparse
import os
import re
import sys
from typing import Dict, List

import pandas as pd


# Block patterns organized by category. Each category has:
#   - "patterns": regex patterns that match off-topic job titles
#   - "keywords": plain strings used to detect overlap with search titles;
#                 if ANY search title contains one of these keywords, the
#                 entire category is skipped.
BLOCK_CATEGORIES: Dict[str, dict] = {
    "healthcare": {
        "keywords": [
            "nurse", "nursing", "rn", "lpn", "cna", "pharmacist", "pharmacy",
            "dental", "dentist", "physician", "surgeon", "medical", "phlebotom",
            "radiolog", "sonograph", "clinical", "healthcare", "health care",
        ],
        "patterns": [
            r"\bnurse\b", r"\bnursing\b", r"\brn\b", r"\blpn\b", r"\bcna\b",
            r"\bpharmacist\b", r"\bpharmacy\b", r"\bdental\b", r"\bdentist\b",
            r"\bphysician\b", r"\bsurgeon\b", r"\bmedical assistant\b",
            r"\bphlebotom", r"\bradiolog", r"\bsonograph",
        ],
    },
    "behavioral_health": {
        "keywords": [
            "behavior", "behavioral", "bcba", "aba", "therapist", "therapy",
        ],
        "patterns": [
            r"\bbehavior tech", r"\bbehavioral tech", r"\bbcba\b", r"\baba therap",
        ],
    },
    "food_service": {
        "keywords": [
            "cook", "chef", "barista", "server", "bartender", "restaurant",
            "food service", "kitchen",
        ],
        "patterns": [
            r"\bcook\b", r"\bchef\b", r"\bbarista\b", r"\bserver\b",
            r"\bbartender\b",
        ],
    },
    "retail": {
        "keywords": [
            "cashier", "retail", "store manager", "store lead", "shift lead",
        ],
        "patterns": [
            r"\bcashier\b",
            r"\bstore lead\b", r"\bstore manager\b", r"\bshift lead\b",
            r"\bmeat\b.*\bseafood\b", r"\bseafood\b.*\bmeat\b",
        ],
    },
    "manual_labor": {
        "keywords": [
            "janitor", "custodian", "housekeeper", "housekeeping",
            "warehouse", "forklift", "loader",
        ],
        "patterns": [
            r"\bjanitor\b", r"\bcustodian\b", r"\bhousekeep",
            r"\bwarehouse\b", r"\bforklift\b", r"\bloader\b",
        ],
    },
    "skilled_trades": {
        "keywords": [
            "electrician", "plumber", "hvac", "welding", "welder",
            "carpenter", "mechanic",
        ],
        "patterns": [
            r"\belectrician\b", r"\bplumber\b", r"\bhvac\b", r"\bwelding\b",
            r"\bcarpenter\b", r"\bmechanic\b",
        ],
    },
    "driving": {
        "keywords": [
            "truck driver", "cdl", "delivery driver", "trucker",
        ],
        "patterns": [
            r"\btruck driver\b", r"\bcdl\b", r"\bdelivery driver\b",
        ],
    },
}


def _build_active_patterns(
    search_titles: List[str],
) -> tuple[list[re.Pattern], list[str], list[str]]:
    """
    Build the compiled regex list, skipping categories that overlap with
    the user's search titles.

    Returns:
        (compiled_patterns, active_category_names, skipped_category_names)
    """
    search_blob = " ".join(t.lower() for t in search_titles)

    active_patterns: list[str] = []
    active_categories: list[str] = []
    skipped_categories: list[str] = []

    for cat_name, cat_info in BLOCK_CATEGORIES.items():
        # Check if any keyword for this category appears in the search titles
        overlap = any(kw in search_blob for kw in cat_info["keywords"])
        if overlap:
            skipped_categories.append(cat_name)
        else:
            active_categories.append(cat_name)
            active_patterns.extend(cat_info["patterns"])

    compiled = [re.compile(p, re.IGNORECASE) for p in active_patterns]
    return compiled, active_categories, skipped_categories


def is_blocked(title: str, patterns: list[re.Pattern]) -> bool:
    """Check if a job title matches any of the given block patterns."""
    for pattern in patterns:
        if pattern.search(title):
            return True
    return False


def filter_relevance(
    df: pd.DataFrame,
    search_titles: List[str],
    title_column: str = "title",
) -> pd.DataFrame:
    """
    Filter a DataFrame of job postings by removing off-topic titles.

    Block-list categories that overlap with the user's search titles are
    automatically disabled, so the filter works for any discipline.

    Args:
        df: DataFrame with job postings
        search_titles: The titles used to search (e.g., ["Economist", "Policy Analyst"])
        title_column: Column containing job titles

    Returns:
        Filtered DataFrame with off-topic postings removed.
    """
    original_count = len(df)

    # Build patterns, skipping categories relevant to the user's search
    patterns, active_cats, skipped_cats = _build_active_patterns(search_titles)

    if skipped_cats:
        print(f"  Filter categories disabled (overlap with search titles): {', '.join(skipped_cats)}")
    print(f"  Active filter categories: {', '.join(active_cats) if active_cats else '(none)'}")

    if not patterns:
        print(f"  Relevance filter: all categories disabled — keeping all {original_count} postings")
        return df.copy()

    def should_block(title):
        if pd.isna(title):
            return False
        return is_blocked(str(title), patterns)

    blocked_mask = df[title_column].apply(should_block)
    blocked_df = df[blocked_mask]
    df_filtered = df[~blocked_mask].copy()

    # Report
    blocked_count = len(blocked_df)
    kept_count = len(df_filtered)

    print(f"  Relevance filter: {original_count} -> {kept_count} postings")
    print(f"    Blocked (off-topic titles):    {blocked_count}")
    print(f"    Kept:                          {kept_count}")

    if blocked_count > 0:
        print(f"    Sample blocked titles:")
        for title in blocked_df[title_column].unique()[:5]:
            print(f"      - {title}")

    return df_filtered


def main():
    """Standalone mode: filter an existing CSV."""
    parser = argparse.ArgumentParser(
        description="UpskillScan Relevance Filter — Remove off-topic job postings"
    )
    parser.add_argument("--input", required=True, help="Path to scraped jobs CSV")
    parser.add_argument("--output", default=None, help="Output CSV path (default: overwrite input)")
    parser.add_argument(
        "--titles", required=True,
        help="Comma-separated search titles used during scraping",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    titles = [t.strip() for t in args.titles.split(",")]
    print(f"Loaded {len(df)} postings from {args.input}")
    print(f"Search titles: {titles}\n")

    df_filtered = filter_relevance(df, search_titles=titles)

    output_path = args.output or args.input.replace(".csv", "_filtered.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"\nSaved filtered data to {output_path}")


if __name__ == "__main__":
    main()

"""
Relevance Filter — Remove off-topic job postings before LLM extraction.

JobSpy returns loosely-matched results (e.g., searching "Economic Associate"
may return Walmart store leads, nurses, cooks). This module filters out
postings whose titles are clearly unrelated to the search intent using
a block-list of known off-topic patterns.

Can be used in the pipeline (main.py) or standalone:
    python relevance_filter.py --input output/raw_jobs.csv --titles "Economist,Policy Analyst"
"""

import argparse
import os
import re
import sys
from typing import List

import pandas as pd


# Job title patterns that are almost never relevant to white-collar analytics/policy roles.
# These catch the most common noise from JobSpy's fuzzy matching.
BLOCK_PATTERNS = [
    # Healthcare / Clinical
    r"\bnurse\b", r"\bnursing\b", r"\brn\b", r"\blpn\b", r"\bcna\b",
    r"\bpharmacist\b", r"\bpharmacy\b", r"\bdental\b", r"\bdentist\b",
    r"\bphysician\b", r"\bsurgeon\b", r"\bmedical assistant\b",
    r"\bphlebotom", r"\bradiolog", r"\bsonograph",
    # Behavioral health (unless user is searching for it)
    r"\bbehavior tech", r"\bbehavioral tech", r"\bbcba\b", r"\baba therap",
    # Food service / Retail / Manual labor
    r"\bcook\b", r"\bchef\b", r"\bbarista\b", r"\bserver\b", r"\bbartender\b",
    r"\bcashier\b", r"\bjanitor\b", r"\bcustodian\b", r"\bhousekeep",
    r"\bmeat\b.*\bseafood\b", r"\bseafood\b.*\bmeat\b",
    r"\bstore lead\b", r"\bstore manager\b", r"\bshift lead\b",
    r"\bwarehouse\b", r"\bforklift\b", r"\bloader\b",
    # Skilled trades
    r"\belectrician\b", r"\bplumber\b", r"\bhvac\b", r"\bwelding\b",
    r"\bcarpenter\b", r"\bmechanic\b",
    # CDL / Driving
    r"\btruck driver\b", r"\bcdl\b", r"\bdelivery driver\b",
]

# Compile once for performance
_BLOCK_RE = [re.compile(p, re.IGNORECASE) for p in BLOCK_PATTERNS]


def is_blocked(title: str) -> bool:
    """Check if a job title matches any block pattern."""
    for pattern in _BLOCK_RE:
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

    Uses a block-list of regex patterns to remove postings whose titles
    match known off-topic categories (healthcare, food service, trades, etc.).

    Args:
        df: DataFrame with job postings
        search_titles: The titles used to search (e.g., ["Economist", "Policy Analyst"])
        title_column: Column containing job titles

    Returns:
        Filtered DataFrame with off-topic postings removed.
    """
    original_count = len(df)

    # Ensure search titles themselves aren't caught by block list
    search_lower = {t.lower() for t in search_titles}

    def should_block(title):
        if pd.isna(title):
            return False
        # Don't block if the title closely matches a search title
        if title.lower().strip() in search_lower:
            return False
        return is_blocked(str(title))

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

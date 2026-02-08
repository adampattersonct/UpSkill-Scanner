"""
Relevance Filter — Remove off-topic job postings before LLM extraction.

JobSpy returns loosely-matched results (e.g., searching "Economic Associate"
may return Walmart store leads, nurses, cooks). This module filters out
postings whose titles are clearly unrelated to the search intent.

Uses a two-pass approach:
  1. Block list — remove titles matching known off-topic patterns
  2. Relevance scoring — check that the job title shares meaningful
     words with at least one of the search titles

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


def title_words(text: str) -> set:
    """Extract meaningful words (3+ chars) from a title, lowercased."""
    words = set(re.findall(r"[a-z]{3,}", text.lower()))
    # Remove very common filler words
    stopwords = {"the", "and", "for", "with", "from", "this", "that", "are", "was",
                 "will", "can", "has", "have", "been", "not", "but", "all", "their",
                 "our", "you", "your", "they", "who", "how", "new", "per", "via",
                 "usa", "remote", "hybrid", "full", "time", "part", "entry", "level",
                 "senior", "junior", "lead", "staff", "associate", "specialist",
                 "coordinator", "manager", "director", "principal", "intern"}
    return words - stopwords


def relevance_score(job_title: str, search_titles: List[str]) -> float:
    """
    Score how relevant a job title is to the search titles.
    Returns 0.0 (no overlap) to 1.0 (strong match).
    """
    job_words = title_words(job_title)
    if not job_words:
        return 0.0

    best_overlap = 0.0
    for search_title in search_titles:
        search_words = title_words(search_title)
        if not search_words:
            continue
        overlap = len(job_words & search_words)
        # Normalize by the smaller set (so "Economist" matches "Senior Economist")
        score = overlap / min(len(job_words), len(search_words)) if overlap > 0 else 0.0
        best_overlap = max(best_overlap, score)

    return best_overlap


def filter_relevance(
    df: pd.DataFrame,
    search_titles: List[str],
    min_score: float = 0.0,
    title_column: str = "title",
) -> pd.DataFrame:
    """
    Filter a DataFrame of job postings for relevance.

    Args:
        df: DataFrame with job postings
        search_titles: The titles used to search (e.g., ["Economist", "Policy Analyst"])
        min_score: Minimum relevance score to keep (0.0 = only block-list filtering)
        title_column: Column containing job titles

    Returns:
        Filtered DataFrame with added 'relevance_score' column.
        Also prints a summary of what was removed.
    """
    original_count = len(df)

    # Ensure search titles themselves aren't caught by block list
    search_lower = {t.lower() for t in search_titles}

    # Pass 1: Block list
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

    # Pass 2: Relevance scoring
    df_filtered["relevance_score"] = df_filtered[title_column].apply(
        lambda t: relevance_score(str(t), search_titles) if pd.notna(t) else 0.0
    )

    if min_score > 0:
        low_relevance = df_filtered[df_filtered["relevance_score"] < min_score]
        df_filtered = df_filtered[df_filtered["relevance_score"] >= min_score]
    else:
        low_relevance = pd.DataFrame()

    # Report
    blocked_count = len(blocked_df)
    low_count = len(low_relevance)
    kept_count = len(df_filtered)

    print(f"  Relevance filter: {original_count} -> {kept_count} postings")
    print(f"    Blocked (off-topic titles):    {blocked_count}")
    if min_score > 0:
        print(f"    Low relevance (< {min_score}):     {low_count}")
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
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Minimum relevance score to keep (0.0 = block-list only, 0.3 = moderate, 0.5 = strict)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    titles = [t.strip() for t in args.titles.split(",")]
    print(f"Loaded {len(df)} postings from {args.input}")
    print(f"Search titles: {titles}\n")

    df_filtered = filter_relevance(df, search_titles=titles, min_score=args.min_score)

    output_path = args.output or args.input.replace(".csv", "_filtered.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"\nSaved filtered data to {output_path}")


if __name__ == "__main__":
    main()

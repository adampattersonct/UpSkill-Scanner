#!/usr/bin/env python3
"""Generate a publication-ready comparison table combining national and regional skill data."""

import argparse
import os

import pandas as pd


def build_publication_table(output_dir: str = "output", top_n: int = 20) -> pd.DataFrame:
    """Build a combined national + regional skill table.

    Args:
        output_dir: Directory containing national_ranking.csv and regional_matrix.csv.
        top_n: Number of top skills to include (by national count).

    Returns:
        DataFrame with top_n rows and columns for national + per-region percentages.
    """
    national = pd.read_csv(os.path.join(output_dir, "national_ranking.csv"))
    regional = pd.read_csv(os.path.join(output_dir, "regional_matrix.csv"))

    # Top N skills by national count
    top_skills = national.head(top_n)[["skill", "category_label", "percentage"]].copy()
    top_skills.columns = ["Skill", "Category", "National (%)"]

    # Join regional percentages
    regional_cols = ["skill", "Northeast", "Southeast", "Midwest", "South_Central", "West"]
    top_skills = top_skills.merge(
        regional[regional_cols], left_on="Skill", right_on="skill", how="left"
    ).drop(columns=["skill"])

    # Rename South_Central for readability
    top_skills.rename(columns={"South_Central": "South Central"}, inplace=True)

    # Round all percentage columns to 1 decimal
    pct_cols = ["National (%)", "Northeast (%)", "Southeast (%)", "Midwest (%)",
                "South Central (%)", "West (%)"]
    # Add (%) suffix to region columns
    top_skills.rename(columns={
        "Northeast": "Northeast (%)",
        "Southeast": "Southeast (%)",
        "Midwest": "Midwest (%)",
        "South Central": "South Central (%)",
        "West": "West (%)",
    }, inplace=True)
    top_skills[pct_cols] = top_skills[pct_cols].round(1)

    # Sort by national percentage descending (should already be, but enforce)
    top_skills.sort_values("National (%)", ascending=False, inplace=True, ignore_index=True)

    return top_skills


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready skill comparison table.")
    parser.add_argument("--output-dir", default="output", help="Directory for input/output files (default: output)")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top skills to include (default: 20)")
    args = parser.parse_args()

    table = build_publication_table(args.output_dir, args.top_n)

    out_path = os.path.join(args.output_dir, "publication_table.csv")
    table.to_csv(out_path, index=False)
    print(f"Saved to {out_path}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

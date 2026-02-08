"""
Statistical Sanity Checks — Validation for UpskillScan
=======================================================
Runs statistical analyses on extracted skills to check for anomalies:
- Skill frequency distribution (Zipf's law test)
- Skills per posting distribution
- Category balance analysis
- Outlier detection (suspiciously rare or common skills)
- Empty/low extraction flagging

Usage:
    python validation/sanity_checks.py
    python validation/sanity_checks.py --input output/extracted_jobs.csv --ranking output/national_ranking.csv
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SKILL_COLUMNS = ["technical_skills", "soft_skills", "certifications", "domain_knowledge"]
CATEGORY_LABELS = {
    "technical_skills": "Technical",
    "soft_skills": "Soft Skills",
    "certifications": "Certifications",
    "domain_knowledge": "Domain Knowledge",
}


def parse_skills(value):
    if pd.isna(value) or value == "[]":
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def analyze_skills_per_posting(df: pd.DataFrame) -> dict:
    """Analyze the distribution of skills per posting."""
    counts = []
    category_counts = {col: [] for col in SKILL_COLUMNS}

    for _, row in df.iterrows():
        total = 0
        for col in SKILL_COLUMNS:
            skills = parse_skills(row.get(col))
            n = len(skills)
            total += n
            category_counts[col].append(n)
        counts.append(total)

    counts = np.array(counts)
    stats = {
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "std": float(np.std(counts)),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "q25": float(np.percentile(counts, 25)),
        "q75": float(np.percentile(counts, 75)),
        "zero_skill_postings": int(np.sum(counts == 0)),
        "low_skill_postings": int(np.sum(counts < 5)),  # suspiciously few
        "high_skill_postings": int(np.sum(counts > 50)),  # suspiciously many
    }

    category_stats = {}
    for col in SKILL_COLUMNS:
        arr = np.array(category_counts[col])
        category_stats[col] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "zero_count": int(np.sum(arr == 0)),
            "pct_with_zero": float(np.sum(arr == 0) / len(arr) * 100),
        }

    return {
        "overall": stats,
        "category": category_stats,
        "raw_counts": counts,
        "category_counts": category_counts,
    }


def analyze_frequency_distribution(ranking_df: pd.DataFrame) -> dict:
    """Check if skill frequencies follow Zipf's law (expected for natural language)."""
    counts = ranking_df["count"].values
    counts_sorted = np.sort(counts)[::-1]
    ranks = np.arange(1, len(counts_sorted) + 1)

    # Zipf's law: frequency ~ 1/rank^alpha
    # In log space: log(freq) = -alpha * log(rank) + C
    # Fit a line in log-log space (only for skills with count > 1)
    mask = counts_sorted > 1
    if mask.sum() > 10:
        log_ranks = np.log(ranks[mask])
        log_freqs = np.log(counts_sorted[mask])
        coeffs = np.polyfit(log_ranks, log_freqs, 1)
        alpha = -coeffs[0]
        r_squared = 1 - np.sum((log_freqs - np.polyval(coeffs, log_ranks)) ** 2) / \
                        np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    else:
        alpha = None
        r_squared = None

    # Count singletons (skills appearing only once)
    singleton_count = int(np.sum(counts == 1))
    singleton_pct = singleton_count / len(counts) * 100

    return {
        "total_unique_skills": len(counts),
        "zipf_alpha": float(alpha) if alpha else None,
        "zipf_r_squared": float(r_squared) if r_squared else None,
        "singleton_skills": singleton_count,
        "singleton_pct": singleton_pct,
        "top_10_concentration": float(counts_sorted[:10].sum() / counts_sorted.sum() * 100),
        "top_50_concentration": float(counts_sorted[:50].sum() / counts_sorted.sum() * 100),
        "counts_sorted": counts_sorted,
        "ranks": ranks,
    }


def detect_outliers(ranking_df: pd.DataFrame) -> dict:
    """Flag skills that seem anomalous."""
    results = {"high_frequency_outliers": [], "category_mismatches": []}

    # Skills with unusually high percentage
    high_pct = ranking_df[ranking_df["percentage"] > 80]
    for _, row in high_pct.iterrows():
        results["high_frequency_outliers"].append({
            "skill": row["skill"],
            "percentage": row["percentage"],
            "category": row["category"],
        })

    # Check for potential category mismatches (soft skills that look technical, etc.)
    # Use word-boundary matching to avoid false positives (e.g., "r" matching inside "leadership")
    import re
    tech_keywords = ["python", "sql", "excel", r"\br\b", "sas", "stata", "tableau", "power bi",
                     "programming", "software", "database", "api", "cloud"]
    soft_keywords = ["communication", "leadership", "teamwork", "interpersonal",
                     "problem solving", "critical thinking", "collaboration"]

    def keyword_match(keywords, text):
        return any(re.search(kw, text) if kw.startswith("\\b") else kw in text for kw in keywords)

    for _, row in ranking_df.iterrows():
        skill_lower = row["skill"].lower()
        cat = row["category"]

        if cat == "soft_skills" and keyword_match(tech_keywords, skill_lower):
            results["category_mismatches"].append({
                "skill": row["skill"],
                "assigned_category": cat,
                "likely_category": "technical_skills",
                "count": row["count"],
            })
        elif cat == "technical_skills" and keyword_match(soft_keywords, skill_lower):
            results["category_mismatches"].append({
                "skill": row["skill"],
                "assigned_category": cat,
                "likely_category": "soft_skills",
                "count": row["count"],
            })

    return results


def generate_plots(posting_stats: dict, freq_stats: dict, ranking_df: pd.DataFrame, output_dir: str):
    """Generate validation charts."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    colors = sns.color_palette("colorblind")

    # 1. Skills per posting histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(posting_stats["raw_counts"], bins=30, color=colors[0], edgecolor="white", alpha=0.8)
    ax.axvline(posting_stats["overall"]["mean"], color=colors[3], linestyle="--",
               label=f'Mean: {posting_stats["overall"]["mean"]:.1f}')
    ax.axvline(posting_stats["overall"]["median"], color=colors[2], linestyle="--",
               label=f'Median: {posting_stats["overall"]["median"]:.0f}')
    ax.set_xlabel("Skills Extracted per Posting", fontsize=12)
    ax.set_ylabel("Number of Postings", fontsize=12)
    ax.set_title("Distribution of Skills per Posting", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "skills_per_posting_hist.png"), dpi=300)
    plt.close()

    # 2. Zipf plot (log-log rank vs frequency)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freq_stats["ranks"], freq_stats["counts_sorted"], "o",
              markersize=2, alpha=0.5, color=colors[0])
    if freq_stats["zipf_alpha"]:
        ax.set_title(
            f"Skill Frequency vs. Rank (Zipf Plot)\n"
            f"alpha={freq_stats['zipf_alpha']:.2f}, R²={freq_stats['zipf_r_squared']:.3f}",
            fontsize=14
        )
    else:
        ax.set_title("Skill Frequency vs. Rank (Zipf Plot)", fontsize=14)
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zipf_plot.png"), dpi=300)
    plt.close()

    # 3. Category balance per posting (box plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_data = []
    cat_labels = []
    for col in SKILL_COLUMNS:
        cat_data.append(posting_stats["category_counts"][col])
        cat_labels.append(CATEGORY_LABELS[col])
    bp = ax.boxplot(cat_data, labels=cat_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:4]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Skills per Posting", fontsize=12)
    ax.set_title("Skill Count Distribution by Category", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_boxplot.png"), dpi=300)
    plt.close()

    # 4. Cumulative concentration curve
    fig, ax = plt.subplots(figsize=(10, 6))
    total = freq_stats["counts_sorted"].sum()
    cumulative = np.cumsum(freq_stats["counts_sorted"]) / total * 100
    n_skills = len(cumulative)
    x_pct = np.arange(1, n_skills + 1) / n_skills * 100
    ax.plot(x_pct, cumulative, color=colors[0], linewidth=2)
    ax.plot([0, 100], [0, 100], "--", color="gray", alpha=0.5, label="Perfect equality")
    ax.set_xlabel("Percentage of Unique Skills (ranked by frequency)", fontsize=12)
    ax.set_ylabel("Cumulative Percentage of Total Mentions", fontsize=12)
    ax.set_title("Skill Concentration Curve", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "concentration_curve.png"), dpi=300)
    plt.close()

    print(f"Plots saved to {output_dir}/")


def print_report(posting_stats: dict, freq_stats: dict, outliers: dict):
    """Print the full sanity check report."""
    print("=" * 70)
    print("STATISTICAL SANITY CHECKS — VALIDATION REPORT")
    print("=" * 70)

    # 1. Skills per posting
    s = posting_stats["overall"]
    print("\n--- Skills per Posting ---")
    print(f"  Mean:       {s['mean']:.1f}")
    print(f"  Median:     {s['median']:.0f}")
    print(f"  Std Dev:    {s['std']:.1f}")
    print(f"  Range:      {s['min']} — {s['max']}")
    print(f"  IQR:        {s['q25']:.0f} — {s['q75']:.0f}")
    print(f"  Zero-skill postings:  {s['zero_skill_postings']}"
          + (" [WARNING]" if s["zero_skill_postings"] > 0 else " [OK]"))
    print(f"  Low-skill (<5):       {s['low_skill_postings']}"
          + (f" [REVIEW]" if s["low_skill_postings"] > 10 else " [OK]"))
    print(f"  High-skill (>50):     {s['high_skill_postings']}"
          + (f" [REVIEW]" if s["high_skill_postings"] > 10 else " [OK]"))

    # 2. Per-category breakdown
    print("\n--- Skills per Posting by Category ---")
    for col in SKILL_COLUMNS:
        cs = posting_stats["category"][col]
        label = CATEGORY_LABELS[col]
        print(f"  {label:25s}: mean={cs['mean']:.1f}  median={cs['median']:.0f}  "
              f"zero_pct={cs['pct_with_zero']:.1f}%")

    # 3. Frequency distribution
    print("\n--- Skill Frequency Distribution ---")
    print(f"  Total unique skills:    {freq_stats['total_unique_skills']:,}")
    print(f"  Singleton skills (n=1): {freq_stats['singleton_skills']:,} ({freq_stats['singleton_pct']:.1f}%)")
    if freq_stats["zipf_alpha"]:
        print(f"  Zipf alpha:             {freq_stats['zipf_alpha']:.2f}"
              + (" [typical range: 0.8–1.5]" if 0.5 < freq_stats["zipf_alpha"] < 2.0 else " [UNUSUAL]"))
        print(f"  Zipf R²:               {freq_stats['zipf_r_squared']:.3f}"
              + (" [good fit]" if freq_stats["zipf_r_squared"] > 0.85 else " [weak fit]"))
    print(f"  Top 10 skills hold:     {freq_stats['top_10_concentration']:.1f}% of all mentions")
    print(f"  Top 50 skills hold:     {freq_stats['top_50_concentration']:.1f}% of all mentions")

    # 4. Outliers
    print("\n--- Outlier Detection ---")
    if outliers["high_frequency_outliers"]:
        print(f"  Skills appearing in >80% of postings ({len(outliers['high_frequency_outliers'])}):")
        for o in outliers["high_frequency_outliers"]:
            print(f"    {o['skill']:40s} {o['percentage']:.1f}%  [{o['category']}]")
    else:
        print("  No skills above 80% frequency.")

    if outliers["category_mismatches"]:
        print(f"\n  Potential category mismatches ({len(outliers['category_mismatches'])}):")
        for m in outliers["category_mismatches"][:10]:
            print(f"    {m['skill']:40s} assigned={m['assigned_category']:20s} likely={m['likely_category']}")
    else:
        print("  No obvious category mismatches detected.")

    # 5. Overall verdict
    print("\n" + "=" * 70)
    print("VERDICT:")
    issues = []
    if posting_stats["overall"]["zero_skill_postings"] > 0:
        issues.append(f"{posting_stats['overall']['zero_skill_postings']} postings with zero skills")
    if freq_stats["singleton_pct"] > 70:
        issues.append(f"High singleton rate ({freq_stats['singleton_pct']:.0f}%) — normalization may need tuning")
    if freq_stats["zipf_r_squared"] and freq_stats["zipf_r_squared"] < 0.7:
        issues.append("Weak Zipf fit — unusual frequency distribution")
    if len(outliers["category_mismatches"]) > 20:
        issues.append(f"{len(outliers['category_mismatches'])} potential category mismatches")

    if not issues:
        print("  All checks PASSED. Distribution looks healthy.")
    else:
        print("  Issues to review:")
        for issue in issues:
            print(f"    - {issue}")
    print("=" * 70)


def save_report(posting_stats: dict, freq_stats: dict, outliers: dict, output_dir: str):
    """Save detailed results."""
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "skills_per_posting": posting_stats["overall"],
        "category_stats": posting_stats["category"],
        "frequency_distribution": {
            "total_unique_skills": freq_stats["total_unique_skills"],
            "zipf_alpha": freq_stats["zipf_alpha"],
            "zipf_r_squared": freq_stats["zipf_r_squared"],
            "singleton_skills": freq_stats["singleton_skills"],
            "singleton_pct": freq_stats["singleton_pct"],
            "top_10_concentration": freq_stats["top_10_concentration"],
            "top_50_concentration": freq_stats["top_50_concentration"],
        },
        "outliers": outliers,
    }

    with open(os.path.join(output_dir, "sanity_check_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/sanity_check_results.json")


def main():
    parser = argparse.ArgumentParser(description="UpskillScan Statistical Sanity Checks")
    parser.add_argument("--input", default="output/extracted_jobs.csv", help="Path to extracted jobs CSV")
    parser.add_argument("--ranking", default="output/national_ranking.csv", help="Path to national ranking CSV")
    parser.add_argument("--output-dir", default="validation/results", help="Directory for output files")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    for path in [args.input, args.ranking]:
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run the main pipeline first.")
            sys.exit(1)

    print(f"Loading data...")
    df = pd.read_csv(args.input)
    ranking_df = pd.read_csv(args.ranking)
    print(f"  Postings: {len(df)}")
    print(f"  Unique skills in ranking: {len(ranking_df)}")

    print("\nRunning analysis...")
    posting_stats = analyze_skills_per_posting(df)
    freq_stats = analyze_frequency_distribution(ranking_df)
    outliers = detect_outliers(ranking_df)

    print_report(posting_stats, freq_stats, outliers)
    save_report(posting_stats, freq_stats, outliers, args.output_dir)

    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(posting_stats, freq_stats, ranking_df, args.output_dir)


if __name__ == "__main__":
    main()

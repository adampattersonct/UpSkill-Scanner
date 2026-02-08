"""
Grounding Check — Automated Validation for UpskillScan
=======================================================
Checks whether each extracted skill can be found in the original job description.
Flags potentially hallucinated skills (extracted but not grounded in text).

Usage:
    python validation/grounding_check.py
    python validation/grounding_check.py --input output/extracted_jobs.csv
    python validation/grounding_check.py --threshold 0.7  # flag postings below 70% grounding
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd


SKILL_COLUMNS = ["technical_skills", "soft_skills", "certifications", "domain_knowledge"]

# Common abbreviation expansions to improve fuzzy matching
EXPANSIONS = {
    "sql": ["structured query language", "sql"],
    "excel": ["excel", "microsoft excel", "ms excel"],
    "r": ["r programming", "r language", " r ", " r,"],
    "python": ["python"],
    "sas": [" sas ", "sas programming"],
    "vba": ["vba", "visual basic"],
    "etl": ["etl", "extract transform load"],
    "crm": ["crm", "customer relationship management"],
    "erp": ["erp", "enterprise resource planning"],
    "kpi": ["kpi", "key performance indicator"],
    "roi": ["roi", "return on investment"],
    "gdp": ["gdp", "gross domestic product"],
    "cpa": ["cpa", "certified public accountant"],
    "cfa": ["cfa", "chartered financial analyst"],
    "ml": ["machine learning", " ml "],
    "ai": ["artificial intelligence", " ai "],
    "nlp": ["natural language processing", " nlp "],
    "bi": ["business intelligence", " bi "],
    "qa": ["quality assurance", " qa "],
}


def parse_skills(value):
    """Parse a JSON-encoded skill list from a CSV cell."""
    if pd.isna(value) or value == "[]":
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def skill_is_grounded(skill: str, description: str) -> dict:
    """
    Check if a skill appears in the job description.

    Returns dict with:
        - grounded: bool
        - match_type: 'exact' | 'substring' | 'expansion' | 'not_found'
        - matched_text: the text that matched (if any)
    """
    desc_lower = description.lower()
    skill_lower = skill.lower().strip()

    # 1. Exact match (case-insensitive)
    if skill_lower in desc_lower:
        return {"grounded": True, "match_type": "exact", "matched_text": skill_lower}

    # 2. Try each word of multi-word skills (at least 2 words must match for skills with 3+ words)
    words = skill_lower.split()
    if len(words) >= 3:
        matches = [w for w in words if len(w) > 2 and w in desc_lower]
        if len(matches) >= 2:
            return {"grounded": True, "match_type": "substring", "matched_text": " + ".join(matches)}
    elif len(words) == 2:
        # For 2-word skills, both words should appear
        if all(w in desc_lower for w in words if len(w) > 2):
            return {"grounded": True, "match_type": "substring", "matched_text": " + ".join(words)}

    # 3. Check known expansions/abbreviations
    for abbrev, expansions in EXPANSIONS.items():
        if skill_lower == abbrev or abbrev in skill_lower:
            for expansion in expansions:
                if expansion in desc_lower:
                    return {"grounded": True, "match_type": "expansion", "matched_text": expansion}

    # 4. Try without common suffixes/prefixes
    for suffix in [" management", " analysis", " skills", " programming"]:
        if skill_lower.endswith(suffix):
            core = skill_lower.replace(suffix, "").strip()
            if len(core) > 2 and core in desc_lower:
                return {"grounded": True, "match_type": "substring", "matched_text": core}

    # 5. Try without hyphens, slashes, parentheses
    cleaned = re.sub(r"[-/().]", " ", skill_lower).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned != skill_lower and cleaned in desc_lower:
        return {"grounded": True, "match_type": "substring", "matched_text": cleaned}

    return {"grounded": False, "match_type": "not_found", "matched_text": None}


def run_grounding_check(df: pd.DataFrame) -> dict:
    """Run grounding check across all postings. Returns detailed results."""
    results = {
        "per_posting": [],
        "ungrounded_skills": defaultdict(list),  # skill -> list of posting indices
        "category_stats": {},
        "match_type_counts": defaultdict(int),
    }

    total_skills = 0
    total_grounded = 0
    category_totals = defaultdict(int)
    category_grounded = defaultdict(int)

    for idx, row in df.iterrows():
        desc = str(row.get("description", ""))
        if not desc or desc == "nan":
            continue

        posting_skills = 0
        posting_grounded = 0
        posting_ungrounded = []

        for col in SKILL_COLUMNS:
            skills = parse_skills(row.get(col))
            for skill in skills:
                total_skills += 1
                posting_skills += 1
                category_totals[col] += 1

                result = skill_is_grounded(skill, desc)
                results["match_type_counts"][result["match_type"]] += 1

                if result["grounded"]:
                    total_grounded += 1
                    posting_grounded += 1
                    category_grounded[col] += 1
                else:
                    posting_ungrounded.append({"skill": skill, "category": col})
                    results["ungrounded_skills"][skill].append(idx)

        grounding_rate = posting_grounded / posting_skills if posting_skills > 0 else 1.0
        results["per_posting"].append({
            "index": idx,
            "title": row.get("title", ""),
            "company": row.get("company", ""),
            "total_skills": posting_skills,
            "grounded": posting_grounded,
            "ungrounded": posting_skills - posting_grounded,
            "grounding_rate": grounding_rate,
            "ungrounded_skills": posting_ungrounded,
        })

    # Overall stats
    results["overall"] = {
        "total_skills_checked": total_skills,
        "total_grounded": total_grounded,
        "total_ungrounded": total_skills - total_grounded,
        "grounding_rate": total_grounded / total_skills if total_skills > 0 else 0,
    }

    # Per-category stats
    for col in SKILL_COLUMNS:
        t = category_totals[col]
        g = category_grounded[col]
        results["category_stats"][col] = {
            "total": t,
            "grounded": g,
            "ungrounded": t - g,
            "grounding_rate": g / t if t > 0 else 0,
        }

    return results


def print_report(results: dict, threshold: float = 0.5):
    """Print a formatted validation report."""
    overall = results["overall"]

    print("=" * 70)
    print("GROUNDING CHECK — VALIDATION REPORT")
    print("=" * 70)
    print()

    # Overall
    print(f"Total skill mentions checked:  {overall['total_skills_checked']:,}")
    print(f"Grounded (found in text):      {overall['total_grounded']:,}")
    print(f"Ungrounded (not found):        {overall['total_ungrounded']:,}")
    print(f"Overall grounding rate:        {overall['grounding_rate']:.1%}")
    print()

    # Match type breakdown
    print("--- Match Type Breakdown ---")
    for mtype, count in sorted(results["match_type_counts"].items(), key=lambda x: -x[1]):
        pct = count / overall["total_skills_checked"] * 100 if overall["total_skills_checked"] > 0 else 0
        print(f"  {mtype:15s}: {count:6,}  ({pct:.1f}%)")
    print()

    # Per-category stats
    print("--- Grounding Rate by Skill Category ---")
    for col, stats in results["category_stats"].items():
        label = col.replace("_", " ").title()
        print(f"  {label:25s}: {stats['grounding_rate']:.1%}  ({stats['grounded']}/{stats['total']})")
    print()

    # Postings below threshold
    low_postings = [p for p in results["per_posting"] if p["grounding_rate"] < threshold]
    print(f"--- Postings Below {threshold:.0%} Grounding Rate ({len(low_postings)} found) ---")
    if low_postings:
        for p in sorted(low_postings, key=lambda x: x["grounding_rate"])[:10]:
            print(f"  Row {p['index']:4d}: {p['grounding_rate']:.0%} "
                  f"({p['grounded']}/{p['total_skills']}) — {p['title'][:40]} @ {str(p['company'])[:25]}")
            for us in p["ungrounded_skills"][:5]:
                print(f"           ? {us['skill']}")
            if len(p["ungrounded_skills"]) > 5:
                print(f"           ... and {len(p['ungrounded_skills']) - 5} more")
    else:
        print("  None — all postings above threshold.")
    print()

    # Most frequently ungrounded skills
    print("--- Most Frequently Ungrounded Skills (potential hallucinations) ---")
    freq_ungrounded = sorted(results["ungrounded_skills"].items(), key=lambda x: -len(x[1]))
    for skill, indices in freq_ungrounded[:20]:
        print(f"  {skill:45s}: ungrounded in {len(indices):3d} postings")
    print()

    print("=" * 70)
    print("INTERPRETATION GUIDE:")
    print("  - Grounding rate > 90%: Strong evidence extraction is faithful")
    print("  - 'exact' matches: Skill text found verbatim in description")
    print("  - 'substring' matches: Key words found (fuzzy but reasonable)")
    print("  - 'not_found': Skill may be inferred/hallucinated — review these")
    print("  - Some 'not_found' may be legitimate (implied skills, acronyms)")
    print("=" * 70)


def save_results(results: dict, output_dir: str):
    """Save detailed results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Save per-posting results
    posting_df = pd.DataFrame(results["per_posting"])
    posting_df = posting_df.drop(columns=["ungrounded_skills"])
    posting_df.to_csv(os.path.join(output_dir, "grounding_per_posting.csv"), index=False)

    # Save ungrounded skills list
    ungrounded_rows = []
    for skill, indices in results["ungrounded_skills"].items():
        ungrounded_rows.append({"skill": skill, "times_ungrounded": len(indices)})
    ungrounded_df = pd.DataFrame(ungrounded_rows)
    ungrounded_df = ungrounded_df.sort_values("times_ungrounded", ascending=False)
    ungrounded_df.to_csv(os.path.join(output_dir, "ungrounded_skills.csv"), index=False)

    # Save summary JSON
    summary = {
        "overall": results["overall"],
        "category_stats": results["category_stats"],
        "match_type_counts": dict(results["match_type_counts"]),
    }
    with open(os.path.join(output_dir, "grounding_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"  - grounding_per_posting.csv  (per-posting grounding rates)")
    print(f"  - ungrounded_skills.csv      (skills not found in text)")
    print(f"  - grounding_summary.json     (overall summary)")


def main():
    parser = argparse.ArgumentParser(description="UpskillScan Grounding Check — Validate extracted skills against source text")
    parser.add_argument("--input", default="output/extracted_jobs.csv", help="Path to extracted jobs CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Flag postings below this grounding rate (default: 0.5)")
    parser.add_argument("--output-dir", default="validation/results", help="Directory for output files")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run the extraction pipeline first.")
        sys.exit(1)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} postings.\n")

    print("Running grounding check (this may take a moment)...")
    results = run_grounding_check(df)

    print_report(results, threshold=args.threshold)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()

"""
Human Annotation Tool — Gold Standard Validation for UpskillScan
=================================================================
Randomly samples job postings and presents them for human review.
The annotator marks each extracted skill as correct or incorrect,
and can add missed skills. Computes precision, recall, and F1.

Usage:
    python validation/human_annotation.py
    python validation/human_annotation.py --n 5 --seed 42
    python validation/human_annotation.py --resume  # continue a previous session
"""

import argparse
import json
import os
import random
import sys
import textwrap

import pandas as pd


SKILL_COLUMNS = ["technical_skills", "soft_skills", "certifications", "domain_knowledge"]
CATEGORY_LABELS = {
    "technical_skills": "Technical Skills",
    "soft_skills": "Soft Skills",
    "certifications": "Certifications",
    "domain_knowledge": "Domain Knowledge",
}
OUTPUT_FILE = "validation/results/human_annotations.json"


def parse_skills(value):
    if pd.isna(value) or value == "[]":
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def wrap_text(text, width=100):
    """Wrap long text for terminal display."""
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if len(line) > width:
            wrapped.extend(textwrap.wrap(line, width=width))
        else:
            wrapped.append(line)
    return "\n".join(wrapped)


def display_posting(row, idx):
    """Display a job posting with its description."""
    print("\n" + "=" * 80)
    print(f"POSTING {idx + 1}")
    print("=" * 80)
    print(f"Title:    {row.get('title', 'N/A')}")
    print(f"Company:  {row.get('company', 'N/A')}")
    print(f"Location: {row.get('location', 'N/A')}")
    print(f"Region:   {row.get('region', 'N/A')}")
    print("-" * 80)
    print("JOB DESCRIPTION:")
    print("-" * 80)

    desc = str(row.get("description", ""))
    # Clean up markdown formatting for readability
    desc = desc.replace("\\-", "-").replace("\\*", "*")
    # Show first 3000 chars (enough context without overwhelming)
    if len(desc) > 3000:
        print(wrap_text(desc[:3000]))
        print(f"\n... [truncated — {len(desc)} total chars]")
    else:
        print(wrap_text(desc))
    print("-" * 80)


def annotate_skills(row):
    """Interactive annotation of extracted skills for one posting."""
    all_skills = []
    for col in SKILL_COLUMNS:
        skills = parse_skills(row.get(col))
        for skill in skills:
            all_skills.append({"skill": skill, "category": col})

    print(f"\nExtracted Skills ({len(all_skills)} total):")
    print("For each skill, type: y=correct, n=incorrect, s=skip")
    print()

    annotations = []
    for i, item in enumerate(all_skills):
        label = CATEGORY_LABELS[item["category"]]
        while True:
            response = input(f"  [{label:18s}] {item['skill']:40s}  (y/n/s): ").strip().lower()
            if response in ("y", "n", "s", "yes", "no", "skip", ""):
                break
            print("    Please enter y, n, or s")

        if response in ("y", "yes", ""):
            verdict = "correct"
        elif response in ("n", "no"):
            verdict = "incorrect"
        else:
            verdict = "skipped"

        annotations.append({
            "skill": item["skill"],
            "category": item["category"],
            "verdict": verdict,
        })

    # Ask about missed skills
    print()
    print("Were any skills MISSING that should have been extracted?")
    print("Enter missed skills one per line (empty line to finish):")
    missed_skills = []
    while True:
        missed = input("  Missed skill (or Enter to finish): ").strip()
        if not missed:
            break
        missed_skills.append(missed)

    return annotations, missed_skills


def compute_metrics(all_annotations: list) -> dict:
    """Compute precision, recall, and F1 across all annotated postings."""
    total_extracted = 0
    total_correct = 0
    total_incorrect = 0
    total_missed = 0
    total_skipped = 0

    category_stats = {col: {"correct": 0, "incorrect": 0, "skipped": 0} for col in SKILL_COLUMNS}

    for entry in all_annotations:
        for ann in entry["annotations"]:
            total_extracted += 1
            if ann["verdict"] == "correct":
                total_correct += 1
                category_stats[ann["category"]]["correct"] += 1
            elif ann["verdict"] == "incorrect":
                total_incorrect += 1
                category_stats[ann["category"]]["incorrect"] += 1
            else:
                total_skipped += 1
                category_stats[ann["category"]]["skipped"] += 1

        total_missed += len(entry["missed_skills"])

    # Precision = correct / (correct + incorrect)  [excluding skipped]
    evaluated = total_correct + total_incorrect
    precision = total_correct / evaluated if evaluated > 0 else 0

    # Recall = correct / (correct + missed)
    recall_denom = total_correct + total_missed
    recall = total_correct / recall_denom if recall_denom > 0 else 0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total_extracted": total_extracted,
        "correct": total_correct,
        "incorrect": total_incorrect,
        "skipped": total_skipped,
        "missed": total_missed,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "postings_annotated": len(all_annotations),
        "category_stats": category_stats,
    }


def print_metrics(metrics: dict):
    """Print a formatted metrics report."""
    print("\n" + "=" * 70)
    print("HUMAN ANNOTATION — VALIDATION METRICS")
    print("=" * 70)
    print()
    print(f"Postings annotated:     {metrics['postings_annotated']}")
    print(f"Total skills evaluated: {metrics['total_extracted']}")
    print(f"  Correct:              {metrics['correct']}")
    print(f"  Incorrect:            {metrics['incorrect']}")
    print(f"  Skipped:              {metrics['skipped']}")
    print(f"  Missed by LLM:       {metrics['missed']}")
    print()
    print(f"  PRECISION:            {metrics['precision']:.1%}  (of extracted skills, how many were correct)")
    print(f"  RECALL:               {metrics['recall']:.1%}  (of all real skills, how many were found)")
    print(f"  F1 SCORE:             {metrics['f1']:.1%}")
    print()

    print("--- By Category ---")
    for col in SKILL_COLUMNS:
        stats = metrics["category_stats"][col]
        label = CATEGORY_LABELS[col]
        total = stats["correct"] + stats["incorrect"]
        cat_precision = stats["correct"] / total if total > 0 else 0
        print(f"  {label:25s}: {cat_precision:.0%} precision ({stats['correct']}/{total})")
    print()
    print("=" * 70)
    print("INTERPRETATION:")
    print("  Precision > 90%: Extraction is highly faithful (few hallucinations)")
    print("  Recall > 80%: LLM catches most skills (few omissions)")
    print("  F1 > 85%: Strong overall performance")
    print("=" * 70)


def save_annotations(all_annotations: list, metrics: dict):
    """Save annotations and metrics to JSON."""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    data = {
        "annotations": all_annotations,
        "metrics": metrics,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nAnnotations saved to {OUTPUT_FILE}")


def load_existing_annotations() -> list:
    """Load annotations from a previous session."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            data = json.load(f)
        return data.get("annotations", [])
    return []


def main():
    parser = argparse.ArgumentParser(description="UpskillScan Human Annotation — Gold Standard Validation")
    parser.add_argument("--input", default="output/extracted_jobs.csv", help="Path to extracted jobs CSV")
    parser.add_argument("--n", type=int, default=10, help="Number of postings to annotate (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--resume", action="store_true", help="Resume a previous annotation session")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run the extraction pipeline first.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} postings from {args.input}")

    # Load or initialize annotations
    if args.resume:
        all_annotations = load_existing_annotations()
        already_done = {a["row_index"] for a in all_annotations}
        print(f"Resuming — {len(all_annotations)} postings already annotated.")
    else:
        all_annotations = []
        already_done = set()

    # Sample postings
    random.seed(args.seed)
    available = [i for i in range(len(df)) if i not in already_done]
    n_remaining = args.n - len(all_annotations)

    if n_remaining <= 0:
        print(f"Already annotated {len(all_annotations)} postings (target: {args.n}).")
        metrics = compute_metrics(all_annotations)
        print_metrics(metrics)
        return

    sample_indices = random.sample(available, min(n_remaining, len(available)))
    print(f"Sampling {len(sample_indices)} postings for annotation.\n")
    print("Instructions:")
    print("  - Read the job description carefully")
    print("  - For each extracted skill, judge if it's mentioned/implied in the text")
    print("  - y = correct (skill is in the posting)")
    print("  - n = incorrect (skill is NOT in the posting — hallucination)")
    print("  - s = skip (unsure)")
    print("  - After skills, add any skills the LLM missed")
    print("  - Press Ctrl+C at any time to save progress and quit\n")

    try:
        for count, row_idx in enumerate(sample_indices):
            row = df.iloc[row_idx]
            display_posting(row, len(all_annotations) + count)

            annotations, missed = annotate_skills(row)

            all_annotations.append({
                "row_index": int(row_idx),
                "title": row.get("title", ""),
                "company": str(row.get("company", "")),
                "annotations": annotations,
                "missed_skills": missed,
            })

            # Show running metrics
            metrics = compute_metrics(all_annotations)
            print(f"\n  Running precision: {metrics['precision']:.0%} | "
                  f"Running recall: {metrics['recall']:.0%} | "
                  f"Postings done: {len(all_annotations)}/{args.n}")

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving progress...")

    # Final metrics and save
    if all_annotations:
        metrics = compute_metrics(all_annotations)
        print_metrics(metrics)
        save_annotations(all_annotations, metrics)
    else:
        print("No annotations recorded.")


if __name__ == "__main__":
    main()

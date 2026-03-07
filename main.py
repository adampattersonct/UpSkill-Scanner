#!/usr/bin/env python3
"""CLI entry point / orchestrator for UpskillScan."""

import argparse
import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from config import (
    DEFAULT_COST_LIMIT_USD,
    DEFAULT_JOB_TITLES,
    DEFAULT_MODEL,
    DEFAULT_N_PER_CITY,
    RECLASSIFY_SKILLS,
    REGIONS,
)
from cost_tracker import CostLimitExceeded, CostTracker
from scraper import JobScraper
from extractor import SkillExtractor
from normalizer import SkillNormalizer
from aggregator import SkillAggregator
from visualizer import SkillVisualizer



def parse_args():
    parser = argparse.ArgumentParser(
        description="UpskillScan: Job Market Skill Intelligence Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py                                          # Full run (economics defaults)
  python main.py --regions Northeast --n-per-city 5       # Test run
  python main.py --titles "Physicist,Lab Technician"       # Different discipline
  python main.py --skip-scrape output/raw_jobs.csv         # Reuse scraped data
  python main.py --resume-batch msgbatch_abc123            # Crash recovery
""",
    )
    parser.add_argument(
        "--titles",
        type=str,
        default=None,
        help="Comma-separated job titles (default: economics titles)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="all",
        help='Comma-separated region names or "all" (default: all)',
    )
    parser.add_argument(
        "--n-per-city",
        type=int,
        default=DEFAULT_N_PER_CITY,
        help=f"Max postings per city per title (default: {DEFAULT_N_PER_CITY})",
    )
    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Comma-separated job board sites (default: from config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Anthropic model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Use synchronous API calls instead of Batch API",
    )
    parser.add_argument(
        "--resume-batch",
        type=str,
        default=None,
        help="Resume polling for an existing batch ID (crash recovery)",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=None,
        help="Override COST_LIMIT_USD from .env",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip cost confirmation prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--skip-scrape",
        type=str,
        default=None,
        help="Path to existing scraped CSV (skip scraping step)",
    )
    parser.add_argument(
        "--skip-extract",
        type=str,
        default=None,
        help="Path to existing extracted CSV (skip LLM extraction)",
    )

    return parser.parse_args()


SKILL_COLUMNS = ["technical_skills", "soft_skills", "certifications", "domain_knowledge"]


def reclassify_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Move misclassified skills between categories based on RECLASSIFY_SKILLS config."""
    if not RECLASSIFY_SKILLS:
        return df

    # Build a lowercase lookup
    remap = {k.lower(): v for k, v in RECLASSIFY_SKILLS.items()}
    moved = 0

    for idx, row in df.iterrows():
        for source_col in SKILL_COLUMNS:
            try:
                skills = json.loads(row.get(source_col, "[]"))
            except (json.JSONDecodeError, TypeError):
                continue

            to_move = []
            for skill in skills:
                target = remap.get(skill.lower())
                if target and target != source_col:
                    to_move.append((skill, target))

            if not to_move:
                continue

            # Remove from source
            for skill, _ in to_move:
                skills.remove(skill)
            df.at[idx, source_col] = json.dumps(skills)

            # Add to target
            for skill, target_col in to_move:
                try:
                    target_skills = json.loads(row.get(target_col, "[]"))
                except (json.JSONDecodeError, TypeError):
                    target_skills = []
                if skill not in target_skills:
                    target_skills.append(skill)
                df.at[idx, target_col] = json.dumps(target_skills)
                moved += 1

    if moved:
        print(f"  Reclassified {moved} skill entries across postings")
    return df


def main():
    load_dotenv()
    args = parse_args()

    # --- Config ---
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key and not args.skip_extract:
        print("ERROR: ANTHROPIC_API_KEY not set. Create a .env file (see .env.example).")
        sys.exit(1)

    model = args.model or os.getenv("DEFAULT_MODEL", DEFAULT_MODEL)
    cost_limit = args.cost_limit or float(
        os.getenv("COST_LIMIT_USD", DEFAULT_COST_LIMIT_USD)
    )
    titles = (
        [t.strip() for t in args.titles.split(",")]
        if args.titles
        else DEFAULT_JOB_TITLES
    )
    use_batch = not args.no_batch
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Parse regions
    if args.regions.lower() == "all":
        region_names = list(REGIONS.keys())
    else:
        region_names = [r.strip() for r in args.regions.split(",")]
        for r in region_names:
            if r not in REGIONS:
                print(f"ERROR: Unknown region '{r}'. Valid: {list(REGIONS.keys())}")
                sys.exit(1)

    print("=" * 60)
    print("UpskillScan — Job Market Skill Intelligence Tool")
    print("=" * 60)
    print(f"Job titles: {titles}")
    print(f"Regions: {region_names}")
    print(f"Max per city: {args.n_per_city}")
    print(f"Model: {model}")
    print(f"Batch API: {use_batch}")
    print(f"Cost limit: ${cost_limit:.2f}")
    print(f"Output: {output_dir}/")
    print()

    # --- Initialize ---
    cost_tracker = CostTracker(model=model, cost_limit=cost_limit, use_batch=use_batch)

    # ===== STEP 1: SCRAPE =====
    if args.skip_scrape:
        print(f"Loading scraped data from {args.skip_scrape}...")
        df = pd.read_csv(args.skip_scrape)
        print(f"  Loaded {len(df)} postings")
    else:
        print("Step 1: Scraping job postings...")
        site_names = [s.strip() for s in args.sites.split(",")] if args.sites else None
        scraper = JobScraper(site_names=site_names)
        df = scraper.scrape_all_regions(
            titles=titles,
            n_per_city=args.n_per_city,
            region_names=region_names,
        )
        if df.empty:
            print("No postings found. Exiting.")
            sys.exit(1)

        # Save raw scrape
        raw_path = os.path.join(output_dir, "raw_jobs.csv")
        df.to_csv(raw_path, index=False)
        print(f"  Saved raw data: {raw_path} ({len(df)} postings)")

    # ===== STEP 2: EXTRACT SKILLS =====
    if args.skip_extract:
        print(f"\nLoading extracted data from {args.skip_extract}...")
        df = pd.read_csv(args.skip_extract)
        print(f"  Loaded {len(df)} postings with skills")
    elif args.resume_batch:
        print(f"\nStep 2: Resuming batch {args.resume_batch}...")
        extractor = SkillExtractor(
            api_key=api_key,
            model=model,
            cost_tracker=cost_tracker,
            output_dir=output_dir,
        )
        df = extractor.resume_batch(args.resume_batch, df)
        cost_tracker.print_actual()
    else:
        print("\nStep 2: Extracting skills via LLM...")
        extractor = SkillExtractor(
            api_key=api_key,
            model=model,
            cost_tracker=cost_tracker,
            output_dir=output_dir,
        )

        # Pre-flight cost estimate
        num_valid = df["description"].notna().sum()
        cost_tracker.print_estimate(num_valid)

        if not args.yes:
            confirm = input("\nProceed? [Y/n] ").strip().lower()
            if confirm == "n":
                print("Aborted by user.")
                sys.exit(0)

        try:
            if use_batch:
                df = extractor.extract_batch(df)
            else:
                df = extractor.extract_sync(df)
        except CostLimitExceeded as e:
            print(f"\n{e}")
            print("Partial results saved to checkpoint.")
            sys.exit(1)

        cost_tracker.print_actual()

        # Save extracted data
        extracted_path = os.path.join(output_dir, "extracted_jobs.csv")
        df.to_csv(extracted_path, index=False)
        print(f"  Saved extracted data: {extracted_path}")

    # ===== STEP 3: NORMALIZE SKILLS =====
    print("\nStep 3: Normalizing skills...")
    if api_key:
        normalizer = SkillNormalizer(
            api_key=api_key,
            model=model,
            cost_tracker=cost_tracker,
            output_dir=output_dir,
        )

        # Pre-flight cost estimate for normalization
        categories = normalizer.collect_unique_skills(df)
        total_unique = sum(len(s) for s in categories.values())
        num_chunks = max(1, (total_unique + 199) // 200)
        cost_tracker.print_estimate(num_chunks)

        if not args.yes and not args.skip_extract:
            confirm = input("\nProceed with normalization? [Y/n] ").strip().lower()
            if confirm == "n":
                print("Skipping normalization.")
            else:
                try:
                    df = normalizer.apply_normalization(df, use_batch=use_batch)
                except CostLimitExceeded as e:
                    print(f"\n{e}")
                    print("Continuing with unnormalized skills.")
        else:
            try:
                df = normalizer.apply_normalization(df, use_batch=use_batch)
            except CostLimitExceeded as e:
                print(f"\n{e}")
                print("Continuing with unnormalized skills.")
    else:
        print("  Skipping normalization (no API key)")

    # Reclassify any misplaced skills (e.g., cloud platforms in certifications)
    df = reclassify_skills(df)

    # Save normalized data
    normalized_path = os.path.join(output_dir, "normalized_jobs.csv")
    df.to_csv(normalized_path, index=False)
    print(f"  Saved normalized data: {normalized_path}")

    # ===== STEP 4: AGGREGATE =====
    print("\nStep 4: Aggregating results...")
    aggregator = SkillAggregator()

    national_df = aggregator.aggregate_national(df)
    region_matrix = aggregator.aggregate_by_region(df)
    category_df = aggregator.aggregate_by_category(df)
    stats = aggregator.compute_statistics(df)

    # Save CSVs
    national_path = os.path.join(output_dir, "national_ranking.csv")
    national_df.to_csv(national_path, index=False)
    print(f"  Saved: {national_path}")

    if not region_matrix.empty:
        region_path = os.path.join(output_dir, "regional_matrix.csv")
        region_matrix.to_csv(region_path)
        print(f"  Saved: {region_path}")

    category_path = os.path.join(output_dir, "category_breakdown.csv")
    category_df.to_csv(category_path, index=False)
    print(f"  Saved: {category_path}")

    stats_path = os.path.join(output_dir, "summary_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # ===== STEP 5: VISUALIZE =====
    print("\nStep 5: Generating visualizations...")
    viz = SkillVisualizer(output_dir=output_dir)

    viz.bar_chart_top_skills(national_df)

    if not region_matrix.empty:
        viz.heatmap_skills_by_region(region_matrix)
        # Per-region charts
        for region in region_names:
            if region in region_matrix.columns:
                viz.bar_chart_by_region(region, national_df, region_matrix)

    viz.treemap_by_category(
        national_df,
        job_titles=titles,
        regions=region_names,
        total_postings=stats["total_postings"],
    )

    # Regional heterogeneity analysis (only when analyzing all regions)
    if args.regions.lower() == "all" and not region_matrix.empty:
        print("\n  Generating regional analysis...")
        for region in region_names:
            region_ranking = aggregator.aggregate_for_region(df, region)
            region_count = stats["postings_by_region"].get(region, 0)
            # Save per-region ranking CSV
            region_ranking.to_csv(
                os.path.join(output_dir, f"regional_ranking_{region.lower()}.csv"),
                index=False,
            )
            # Generate per-region treemap
            viz.treemap_by_category(
                region_ranking,
                filename=f"skills_treemap_{region.lower()}.png",
                job_titles=titles,
                regions=[region],
                total_postings=region_count,
            )
        print(f"  Generated {len(region_names)} regional treemaps + ranking CSVs")

    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total postings analyzed: {stats['total_postings']}")
    print(f"Postings with skills: {stats['postings_with_skills']} ({stats['extraction_rate']}%)")
    print(f"Unique skills: {stats['unique_skills']}")
    print(f"Avg skills per posting: {stats['avg_skills_per_posting']}")
    if stats["postings_by_region"]:
        print(f"Postings by region:")
        for region, count in sorted(stats["postings_by_region"].items()):
            print(f"  {region}: {count}")

    if api_key:
        cost_tracker.print_actual()

    print(f"\nAll outputs saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()

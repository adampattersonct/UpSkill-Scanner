#!/usr/bin/env python3
"""
=============================================================
  UpskillScan — Easy Configuration File
=============================================================
  Edit the settings below, then run:
      python run.py
=============================================================
"""
import os

# ──────────────────────────────────────────────────────────
#  JOB TITLES TO SEARCH
#  Change these to match your discipline / field of study
# ──────────────────────────────────────────────────────────
JOB_TITLES = [
    "Junior Economist",
    "Economic Analyst",
    "Financial Analyst",
    "Research Analyst",
    "Data Analyst",
    "Policy Analyst",
]

# Examples for other fields (uncomment one or write your own):
#
# JOB_TITLES = ["Software Engineer", "Backend Developer", "Data Engineer", "ML Engineer"]
# JOB_TITLES = ["Registered Nurse", "Clinical Nurse", "Nurse Practitioner", "Staff Nurse"]
# JOB_TITLES = ["Physicist", "Research Scientist", "Lab Technician", "Physics Engineer"]
# JOB_TITLES = ["Marketing Analyst", "Brand Manager", "Digital Marketing Specialist"]
# JOB_TITLES = ["Mechanical Engineer", "Design Engineer", "Manufacturing Engineer"]

# ──────────────────────────────────────────────────────────
#  REGIONS TO SEARCH
#  Options: "Northeast", "Southeast", "Midwest",
#           "South_Central", "West", or "all"
# ──────────────────────────────────────────────────────────
REGIONS = "all"

# Examples:
# REGIONS = "Northeast"
# REGIONS = "Northeast, West"
# REGIONS = "all"

# ──────────────────────────────────────────────────────────
#  MAX POSTINGS PER CITY
#  Higher = more data but slower scraping ; more costly to run through the model. Adjust based on your needs and budget.
#  Recommended: 5 for testing, 25 for full runs
# ──────────────────────────────────────────────────────────
POSTINGS_PER_CITY = 25
SITE_NAMES = [
    "indeed",
    "glassdoor",
]  # Job board sources (options: "indeed", "linkedin", "zip_recruiter", "glassdoor")

# ──────────────────────────────────────────────────────────
#  AI MODEL
#  "claude-haiku-4-5-20251001"   — cheapest, works great  ($0.50/$2.50 per MTok batch)
#  "claude-sonnet-4-5-20250929"  — smarter, costs more    ($1.50/$7.50 per MTok batch)
# ──────────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5-20251001"

# ──────────────────────────────────────────────────────────
#  COST LIMIT (USD)
#  The tool will stop if estimated spending exceeds this.
#  A full 5-region run with Haiku costs ~$1-2.
# ──────────────────────────────────────────────────────────
COST_LIMIT = 5.00

# ──────────────────────────────────────────────────────────
#  SKIP EXPENSIVE STEPS (for re-running analysis)
#  Set to a file path to reuse previous data, or None to
#  run from scratch.
# ──────────────────────────────────────────────────────────
SKIP_SCRAPE = None  # e.g. "output/raw_jobs.csv"
SKIP_EXTRACT = None  # e.g. "output/normalized_jobs.csv"

# ──────────────────────────────────────────────────────────
#  RUN NAME (for organizing multiple runs)
#  Set a name to save results in output/<run_name>/
#  Set to None to save directly in output/
# ──────────────────────────────────────────────────────────
RUN_NAME = None  # e.g. "economics_northeast", "nursing_all_regions"

# Examples:
# RUN_NAME = "economics_northeast"
# RUN_NAME = "economics_all_25"
# RUN_NAME = "nursing_southeast_test"
# RUN_NAME = "cs_full_run"

# ──────────────────────────────────────────────────────────
#  OTHER OPTIONS
# ──────────────────────────────────────────────────────────
USE_BATCH_API = True  # True = 50% cheaper (recommended), False = slower debug mode
AUTO_CONFIRM = True  # True = skip cost prompts, False = ask before each API call
FILTER_OFF_TOPIC = True  # True = remove off-topic postings before LLM extraction
OUTPUT_DIR = os.path.join("output", RUN_NAME) if RUN_NAME else "output"


# ══════════════════════════════════════════════════════════
#  DO NOT EDIT BELOW THIS LINE
# ══════════════════════════════════════════════════════════

import sys


def build_command():
    """Convert the settings above into a main.py command."""
    cmd = [sys.executable, "main.py"]

    cmd.extend(["--titles", ",".join(JOB_TITLES)])
    cmd.extend(["--regions", REGIONS])
    cmd.extend(["--n-per-city", str(POSTINGS_PER_CITY)])
    cmd.extend(["--sites", ",".join(SITE_NAMES)])
    cmd.extend(["--model", MODEL])
    cmd.extend(["--cost-limit", str(COST_LIMIT)])
    cmd.extend(["--output-dir", OUTPUT_DIR])

    if not USE_BATCH_API:
        cmd.append("--no-batch")
    if AUTO_CONFIRM:
        cmd.append("--yes")
    if not FILTER_OFF_TOPIC:
        cmd.append("--no-filter")
    if SKIP_SCRAPE:
        cmd.extend(["--skip-scrape", SKIP_SCRAPE])
    if SKIP_EXTRACT:
        cmd.extend(["--skip-extract", SKIP_EXTRACT])

    return cmd


if __name__ == "__main__":
    cmd = build_command()

    print("=" * 60)
    print("  UpskillScan — Starting Run")
    print("=" * 60)
    print(f"  Titles:    {JOB_TITLES}")
    print(f"  Regions:   {REGIONS}")
    print(f"  Per city:  {POSTINGS_PER_CITY}")
    print(f"  Model:     {MODEL}")
    print(f"  Cost cap:  ${COST_LIMIT:.2f}")
    print(f"  Batch API: {USE_BATCH_API}")
    print(f"  Output:    {OUTPUT_DIR}/")
    if SKIP_SCRAPE:
        print(f"  Skip scrape: {SKIP_SCRAPE}")
    if SKIP_EXTRACT:
        print(f"  Skip extract: {SKIP_EXTRACT}")
    print("=" * 60)
    print()

    os.execvp(cmd[0], cmd)

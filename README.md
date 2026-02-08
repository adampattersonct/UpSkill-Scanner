# UpskillScan

**Job Market Skill Intelligence Tool** — Scans job postings to identify in-demand skills for any discipline.

UpskillScan scrapes entry-level job postings across 5 US regions (50 metro areas), uses Claude to extract and normalize skills, and produces publication-ready visualizations.

## Quick Start

```bash
# 1. Create a virtual environment (requires Python 3.10+)
python3.12 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
cp .env.example .env
# Edit .env with your Anthropic API key (get one at console.anthropic.com)

# 4. Run
python run.py
```

## Usage

### Easy Mode (Recommended) — `run.py`

Open `run.py` and edit the settings at the top:

```python
JOB_TITLES = [
    "Entry level Economist",
    "Economic Analyst",
    "Data Analyst Economics",
    "Policy Analyst",
    "Research Analyst"
]

REGIONS = "all"           # "Northeast", "West", "all", etc.
POSTINGS_PER_CITY = 25    # 5 for testing, 25 for full runs
COST_LIMIT = 5.00         # Budget cap in USD
```

Then run:
```bash
python run.py
```

That's it. All settings are clearly labeled with examples for different disciplines.

### Advanced Mode — `main.py` with CLI flags

```bash
# Full run with default economics titles
python main.py

# Any discipline — just change --titles
python main.py --titles "Software Engineer,Backend Developer,Data Engineer"
python main.py --titles "Registered Nurse,Clinical Nurse,Nurse Practitioner"

# Control scope
python main.py --titles "Physicist,Lab Technician" --regions Northeast,West --n-per-city 10

# Skip expensive steps (reuse previous data)
python main.py --skip-scrape output/raw_jobs.csv
python main.py --skip-scrape output/raw_jobs.csv --skip-extract output/normalized_jobs.csv

# Crash recovery
python main.py --resume-batch msgbatch_abc123

# Debug mode (synchronous, no batch)
python main.py --regions Northeast --n-per-city 3 --no-batch
```

### CLI Flags Reference

| Flag | What it does |
|---|---|
| `--titles "A,B,C"` | Job titles to search |
| `--regions Northeast` | Which regions (or `all`) |
| `--n-per-city 25` | Max postings per city |
| `--model <model-id>` | Claude model to use |
| `--cost-limit 10.00` | Override budget cap |
| `--yes` | Skip confirmation prompts |
| `--no-batch` | Synchronous mode (slower, for debugging) |
| `--skip-scrape file.csv` | Reuse previously scraped data |
| `--skip-extract file.csv` | Reuse previously extracted data |
| `--resume-batch <id>` | Resume an interrupted batch |
| `--output-dir path/` | Custom output directory |

## Cost Safety

Three layers of protection:
1. **Anthropic Console**: Turn OFF auto-reload — prepaid credits act as a hard cap
2. **Software limit**: `COST_LIMIT` in `run.py` (or `COST_LIMIT_USD` in `.env`) — tool aborts if exceeded
3. **Pre-flight estimates**: Printed before each API batch

Expected cost: ~$1-1.50 for a full 5-region run using Haiku + Batch API.

## Outputs

All saved to `output/`:

| File | Description |
|---|---|
| `raw_jobs.csv` | Scraped postings |
| `extracted_jobs.csv` | Postings with extracted skills |
| `normalized_jobs.csv` | Skills after synonym normalization |
| `national_ranking.csv` | Top skills ranked by frequency |
| `regional_matrix.csv` | Skills x Region percentage matrix |
| `summary_stats.json` | Summary statistics |
| `skill_mapping.json` | Raw-to-canonical skill mapping |
| `top_skills_bar.png` | Bar chart of top 20 skills |
| `skills_region_heatmap.png` | Heatmap of skills by region |
| `skills_treemap.png` | Treemap by skill category |
| `top_skills_<region>.png` | Per-region bar charts |

## Project Structure

```
├── run.py               # Easy config & run (start here)
├── main.py              # CLI entry point with flags
├── config.py            # Regions, defaults, prompts
├── cost_tracker.py      # Budget tracking & limits
├── scraper.py           # JobSpy wrapper
├── extractor.py         # Claude Batch API skill extraction
├── normalizer.py        # Skill synonym grouping
├── aggregator.py        # Counting & ranking
├── visualizer.py        # Chart generation
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
└── output/              # All results saved here
```

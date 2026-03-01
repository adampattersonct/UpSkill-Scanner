# Upskill-Scanner

**Job Market Skill Intelligence Tool** — Scans job postings to identify in-demand skills for any discipline.

Upskill-Scanner scrapes job postings across 5 US regions (50 metro areas), uses Claude to extract and normalize skills, and produces publication-ready visualizations. Job title(s), region, and sample size are inputs of the package. Outputs include visualization of salient and in-demand skills and .csv output for the user to conduct further analysis. 

⚠️ Disclaimer
UpskillScan is provided for educational and academic research purposes only. It depends on JobSpy, which collects publicly available job postings from various job board platforms. The author of Upskill-Scanner are not affiliated with or endorsed by JobSpy or any job board platform. Users are solely responsible for ensuring their use complies with all applicable laws and the terms of service of any platforms from which data is collected. The author is not responsible for any misuse of this software.

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
SITE_NAMES = ["", ""]     # Job board sources (required — no default; input lowercase and all one word ; example indeed , glassdoor)
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
| `--sites "a,b"` | Job board sources to scrape (required) |
| `--no-batch` | Synchronous mode (slower, for debugging) |
| `--no-filter` | Skip relevance filtering of scraped postings |
| `--skip-scrape file.csv` | Reuse previously scraped data |
| `--skip-extract file.csv` | Reuse previously extracted data |
| `--resume-batch <id>` | Resume an interrupted batch |
| `--output-dir path/` | Custom output directory |

## Pipeline

```
JobSpy Scraper → Relevance Filter → Claude Batch API Extraction → Normalization → Aggregation → Visualizations
```

1. **Scrape** — Collects job postings from public job boards via `python-jobspy` (you must specify which sites to scrape in `run.py` or via `--sites`)
2. **Filter** — Removes off-topic postings (e.g., nurses, cooks returned for "Economic Associate") using title-based relevance filtering
3. **Extract** — Sends each description to Claude (Haiku) via the Batch API to extract structured skills into 4 categories: technical, soft, certifications, domain knowledge
4. **Reclassify** — Post-extraction fix for common misclassifications (e.g., cloud platforms like AWS moved from certifications to technical skills)
5. **Normalize** — Groups synonyms via LLM (e.g., "MS Excel" / "Microsoft Excel" → "Excel")
6. **Aggregate** — Counts skill frequency nationally and by region
7. **Visualize** — Generates publication-ready bar charts, heatmaps, and treemaps

## Validation

The `validation/` folder contains three scripts for verifying extraction quality:

```bash
# Automated: checks if extracted skills appear in the source text
python validation/grounding_check.py

# Interactive: manually review 10 sampled postings for precision/recall/F1
python validation/human_annotation.py --n 10

# Statistical: Zipf distribution, outlier detection, category balance
python validation/sanity_checks.py
```

| Script | What it checks | Mode |
|---|---|---|
| `grounding_check.py` | Are extracted skills actually in the job description? | Automated |
| `human_annotation.py` | Precision, recall, F1 via human review | Interactive |
| `sanity_checks.py` | Frequency distributions, outliers, category balance | Automated |

Validation results are saved to `validation/results/`.

## Cost Safety

Three layers of protection:
1. **Anthropic Console**: Turn OFF auto-reload — prepaid credits act as a hard cap
2. **Software limit**: `COST_LIMIT` in `run.py` (or `COST_LIMIT_USD` in `.env`) — tool aborts if exceeded
3. **Pre-flight estimates**: Printed before each API batch

Expected cost: ~$0.70 for a full region run using Haiku + Batch API.

## Outputs

All saved to `output/`:

| File | Description |
|---|---|
| `raw_jobs.csv` | Scraped postings |
| `filtered_jobs.csv` | Postings after relevance filtering |
| `extracted_jobs.csv` | Postings with extracted skills |
| `normalized_jobs.csv` | Skills after synonym normalization |
| `national_ranking.csv` | Top skills ranked by frequency |
| `regional_matrix.csv` | Skills x Region percentage matrix |
| `category_breakdown.csv` | Skill counts by category |
| `summary_stats.json` | Summary statistics |
| `skill_mapping.json` | Raw-to-canonical skill mapping |
| `top_skills_bar.png` | Bar chart of top 20 skills |
| `skills_region_heatmap.png` | Heatmap of skills by region |
| `skills_treemap.png` | Treemap by skill category |
| `top_skills_<region>.png` | Per-region bar charts |
| `skills_treemap_<region>.png` | Per-region treemaps (only when `REGIONS = "all"`) |
| `regional_ranking_<region>.csv` | Per-region skill rankings (only when `REGIONS = "all"`) |

### Regional Heterogeneity Analysis

When `REGIONS = "all"` (default), UpskillScan automatically generates per-region treemaps and ranking CSVs in addition to the national outputs. This enables side-by-side comparison of skill demand across regions — useful for identifying geographic variation in employer expectations.

These additional outputs are only produced when all 5 regions are analyzed, since cross-region comparison requires the full dataset. When running a single region (e.g., `REGIONS = "Northeast"`), only the national-level visualizations are generated.

## Project Structure

```
├── run.py               # Easy config & run (start here)
├── main.py              # CLI entry point with flags
├── config.py            # Regions, defaults, prompts, reclassification rules
├── cost_tracker.py      # Budget tracking & limits
├── scraper.py           # JobSpy wrapper
├── relevance_filter.py  # Post-scrape off-topic filtering
├── extractor.py         # Claude Batch API skill extraction
├── normalizer.py        # Skill synonym grouping
├── aggregator.py        # Counting & ranking
├── visualizer.py        # Chart generation
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
├── validation/          # Validation scripts
│   ├── grounding_check.py
│   ├── human_annotation.py
│   └── sanity_checks.py
└── output/              # All results saved here
```

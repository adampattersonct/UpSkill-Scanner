"""Region definitions, defaults, and constants for UpskillScan."""

REGIONS = {
    "Northeast": [
        "New York, NY", "Boston, MA", "Philadelphia, PA", "Pittsburgh, PA",
        "Hartford, CT", "Providence, RI", "Newark, NJ", "Baltimore, MD",
        "Albany, NY", "Buffalo, NY"
    ],
    "Southeast": [
        "Atlanta, GA", "Miami, FL", "Charlotte, NC", "Nashville, TN",
        "Jacksonville, FL", "Tampa, FL", "Raleigh, NC", "Richmond, VA",
        "Orlando, FL", "Charleston, SC"
    ],
    "Midwest": [
        "Chicago, IL", "Detroit, MI", "Minneapolis, MN", "St. Louis, MO",
        "Indianapolis, IN", "Columbus, OH", "Milwaukee, WI", "Kansas City, MO",
        "Cincinnati, OH", "Cleveland, OH"
    ],
    "South_Central": [
        "Houston, TX", "Dallas, TX", "San Antonio, TX", "Austin, TX",
        "New Orleans, LA", "Memphis, TN", "Oklahoma City, OK",
        "Louisville, KY", "Birmingham, AL", "Little Rock, AR"
    ],
    "West": [
        "Los Angeles, CA", "San Francisco, CA", "Seattle, WA", "Denver, CO",
        "Phoenix, AZ", "Portland, OR", "Las Vegas, NV", "San Diego, CA",
        "Salt Lake City, UT", "Sacramento, CA"
    ],
}

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_COST_LIMIT_USD = 5.00
DEFAULT_JOB_TITLES = [
    "Economist",
    "Economic Analyst",
    "Economic Associate",
    "Data Analyst Economics",
    "Policy Analyst",
]
DEFAULT_N_PER_CITY = 25
MAX_DESCRIPTION_CHARS = 4000
SCRAPE_HOURS_OLD = 168  # 1 week

EXTRACTION_PROMPT = """You are an expert labor market analyst. Extract professional skills from this job posting.

RULES:
- Normalize skill names (e.g., "MS Excel" -> "Excel", "Structured Query Language" -> "SQL")
- Use title case for all skills (e.g., "Python", "Data Visualization")
- Only include skills explicitly mentioned or clearly implied
- Do NOT infer skills that aren't in the text

Return ONLY valid JSON matching this schema:
{{
  "technical_skills": ["string"],
  "soft_skills": ["string"],
  "certifications": ["string"],
  "domain_knowledge": ["string"]
}}

Job Posting:
{description}"""

NORMALIZATION_PROMPT = """You are a labor market taxonomy expert. Below is a list of skills extracted from job postings.
Group them into canonical categories. Many are duplicates or synonyms.

RULES:
- Map each skill to a single canonical name
- Use the most common/recognizable form (e.g., "Excel" not "Microsoft Office Excel")
- Merge obvious synonyms (e.g., "R Programming" and "R Language" -> "R")
- Keep distinct skills separate (don't merge "Python" with "R")

Return a JSON object where keys are the original skill names and values are the canonical names.

Skills to normalize:
{skills_list}"""

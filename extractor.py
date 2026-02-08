"""Anthropic Claude Batch API skill extraction for UpskillScan."""

import json
import os
import time
from typing import Dict, List, Optional

import anthropic
import pandas as pd
from pydantic import BaseModel, Field

from config import DEFAULT_MODEL, EXTRACTION_PROMPT, MAX_DESCRIPTION_CHARS
from cost_tracker import CostLimitExceeded, CostTracker


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` markdown fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text


class SkillExtraction(BaseModel):
    """Structured skill extraction result from a single job posting."""
    technical_skills: List[str] = Field(
        default_factory=list,
        description="Hard/technical skills: programming languages, software tools, statistical methods, platforms",
    )
    soft_skills: List[str] = Field(
        default_factory=list,
        description="Interpersonal competencies: communication, leadership, teamwork, problem-solving",
    )
    certifications: List[str] = Field(
        default_factory=list,
        description="Specific licenses, certifications, or degrees mentioned as requirements",
    )
    domain_knowledge: List[str] = Field(
        default_factory=list,
        description="Subject matter expertise: econometrics, financial modeling, policy analysis, etc.",
    )


class SkillExtractor:
    """Extracts structured skills from job posting descriptions using Claude."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        cost_tracker: CostTracker = None,
        output_dir: str = "output",
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.cost_tracker = cost_tracker
        self.output_dir = output_dir

    def extract_skills_single(self, description: str) -> Optional[SkillExtraction]:
        """Synchronous single-posting extraction (for testing/debugging)."""
        prompt = EXTRACTION_PROMPT.format(
            description=description[:MAX_DESCRIPTION_CHARS]
        )
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                if self.cost_tracker:
                    self.cost_tracker.update_from_usage(
                        response.usage.input_tokens, response.usage.output_tokens
                    )
                    self.cost_tracker.check_limit()

                raw_json = _strip_markdown_fences(response.content[0].text)
                return SkillExtraction.model_validate_json(raw_json)
            except CostLimitExceeded:
                raise
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"  Extraction error (attempt {attempt + 1}/3): {e}")
                if attempt == 2:
                    return None
                time.sleep(1)
        return None

    def extract_sync(self, df: pd.DataFrame, desc_column: str = "description") -> pd.DataFrame:
        """Synchronous one-by-one extraction (--no-batch mode)."""
        total = len(df)
        results: Dict[int, SkillExtraction] = {}

        for i, (idx, row) in enumerate(df.iterrows()):
            if pd.isna(row[desc_column]):
                continue

            print(f"  Extracting {i + 1}/{total}...")
            extraction = self.extract_skills_single(row[desc_column])
            if extraction:
                results[idx] = extraction

            # Checkpoint every 50 rows
            if (i + 1) % 50 == 0:
                self._save_checkpoint(df, results)

        return self._merge_results(df, results)

    def extract_batch(self, df: pd.DataFrame, desc_column: str = "description") -> pd.DataFrame:
        """Batch API extraction — primary mode."""
        requests = self._build_batch_requests(df, desc_column)
        if not requests:
            print("No valid descriptions to process.")
            return df

        BATCH_LIMIT = 10_000
        all_results: Dict[str, SkillExtraction] = {}

        for i in range(0, len(requests), BATCH_LIMIT):
            chunk = requests[i : i + BATCH_LIMIT]

            batch = self.client.messages.batches.create(requests=chunk)
            print(f"  Batch submitted: {batch.id} ({len(chunk)} requests)")

            # Save batch ID immediately for crash recovery
            batch_file = os.path.join(self.output_dir, "batch_ids.txt")
            with open(batch_file, "a") as f:
                f.write(f"{batch.id}\n")

            self._poll_batch(batch.id)
            chunk_results = self._retrieve_batch_results(batch.id)
            all_results.update(chunk_results)

        # Map custom_id back to DataFrame index
        indexed_results: Dict[int, SkillExtraction] = {}
        for custom_id, extraction in all_results.items():
            # custom_id format: "job-{idx}"
            idx = int(custom_id.split("-", 1)[1])
            indexed_results[idx] = extraction

        return self._merge_results(df, indexed_results)

    def resume_batch(self, batch_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """Resume polling and retrieval for an existing batch (crash recovery)."""
        print(f"  Resuming batch: {batch_id}")
        self._poll_batch(batch_id)
        all_results = self._retrieve_batch_results(batch_id)

        indexed_results: Dict[int, SkillExtraction] = {}
        for custom_id, extraction in all_results.items():
            idx = int(custom_id.split("-", 1)[1])
            indexed_results[idx] = extraction

        return self._merge_results(df, indexed_results)

    def _build_batch_requests(self, df: pd.DataFrame, desc_column: str) -> List[dict]:
        """Build batch request objects, one per job posting."""
        requests = []
        for idx, row in df.iterrows():
            if pd.isna(row[desc_column]):
                continue
            prompt = EXTRACTION_PROMPT.format(
                description=row[desc_column][:MAX_DESCRIPTION_CHARS]
            )
            requests.append(
                {
                    "custom_id": f"job-{idx}",
                    "params": {
                        "model": self.model,
                        "max_tokens": 1000,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
            )
        return requests

    def _poll_batch(self, batch_id: str, poll_interval: int = 30) -> None:
        """Poll until batch processing ends."""
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            print(
                f"  Processing: {counts.processing} | "
                f"Succeeded: {counts.succeeded} | "
                f"Errored: {counts.errored}"
            )
            if batch.processing_status == "ended":
                break
            time.sleep(poll_interval)

    def _retrieve_batch_results(self, batch_id: str) -> Dict[str, SkillExtraction]:
        """Retrieve and parse results from a completed batch."""
        results: Dict[str, SkillExtraction] = {}
        for entry in self.client.messages.batches.results(batch_id):
            if entry.result.type == "succeeded":
                # Track costs
                if self.cost_tracker:
                    usage = entry.result.message.usage
                    self.cost_tracker.update_from_usage(
                        usage.input_tokens, usage.output_tokens
                    )

                raw_json = _strip_markdown_fences(entry.result.message.content[0].text)
                try:
                    skills = SkillExtraction.model_validate_json(raw_json)
                    results[entry.custom_id] = skills
                except Exception as e:
                    print(f"  Parse error for {entry.custom_id}: {e}")
            else:
                print(f"  Failed: {entry.custom_id} — {entry.result.type}")
        return results

    def _merge_results(
        self, df: pd.DataFrame, results: Dict[int, SkillExtraction]
    ) -> pd.DataFrame:
        """Merge extraction results back into the DataFrame."""
        df = df.copy()
        df["technical_skills"] = None
        df["soft_skills"] = None
        df["certifications"] = None
        df["domain_knowledge"] = None

        for idx, extraction in results.items():
            if idx in df.index:
                df.at[idx, "technical_skills"] = json.dumps(extraction.technical_skills)
                df.at[idx, "soft_skills"] = json.dumps(extraction.soft_skills)
                df.at[idx, "certifications"] = json.dumps(extraction.certifications)
                df.at[idx, "domain_knowledge"] = json.dumps(extraction.domain_knowledge)

        extracted = len(results)
        total = len(df)
        print(f"  Extracted skills from {extracted}/{total} postings")
        return df

    def _save_checkpoint(
        self, df: pd.DataFrame, results: Dict[int, SkillExtraction]
    ) -> None:
        """Save intermediate results to checkpoint file."""
        checkpoint_df = self._merge_results(df, results)
        path = os.path.join(self.output_dir, "checkpoint.csv")
        checkpoint_df.to_csv(path, index=False)
        print(f"  Checkpoint saved: {path} ({len(results)} extracted)")

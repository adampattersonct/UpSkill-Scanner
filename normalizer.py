"""Skill normalization / synonym grouping for UpskillScan."""

import json
import os
import time
from typing import Dict, List, Set

import anthropic
import pandas as pd

from config import DEFAULT_MODEL, NORMALIZATION_PROMPT
from cost_tracker import CostLimitExceeded, CostTracker
from extractor import _strip_markdown_fences


class SkillNormalizer:
    """Groups synonym skills into canonical names using Claude."""

    CHUNK_SIZE = 200  # Max skills per normalization request

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

    def collect_unique_skills(self, df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Collect all unique skills from the DataFrame, grouped by category."""
        categories = {
            "technical_skills": set(),
            "soft_skills": set(),
            "certifications": set(),
            "domain_knowledge": set(),
        }
        for col in categories:
            if col not in df.columns:
                continue
            for val in df[col].dropna():
                try:
                    skills = json.loads(val) if isinstance(val, str) else val
                    if isinstance(skills, list):
                        categories[col].update(skills)
                except (json.JSONDecodeError, TypeError):
                    pass
        return categories

    def normalize_skills(
        self, raw_skills: List[str], use_batch: bool = True
    ) -> Dict[str, str]:
        """Send skills to Claude for canonical grouping. Returns raw->canonical mapping."""
        if not raw_skills:
            return {}

        # Chunk into groups of CHUNK_SIZE
        chunks = [
            raw_skills[i : i + self.CHUNK_SIZE]
            for i in range(0, len(raw_skills), self.CHUNK_SIZE)
        ]

        if use_batch and len(chunks) > 1:
            return self._normalize_batch(chunks)
        else:
            return self._normalize_sync(chunks)

    def _normalize_sync(self, chunks: List[List[str]]) -> Dict[str, str]:
        """Normalize skill chunks synchronously."""
        mapping: Dict[str, str] = {}
        for i, chunk in enumerate(chunks):
            print(f"  Normalizing chunk {i + 1}/{len(chunks)} ({len(chunk)} skills)...")
            prompt = NORMALIZATION_PROMPT.format(skills_list=json.dumps(chunk, indent=2))

            for attempt in range(3):
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    if self.cost_tracker:
                        self.cost_tracker.update_from_usage(
                            response.usage.input_tokens, response.usage.output_tokens
                        )
                        self.cost_tracker.check_limit()

                    raw_text = _strip_markdown_fences(response.content[0].text)
                    chunk_mapping = json.loads(raw_text)
                    mapping.update(chunk_mapping)
                    break
                except CostLimitExceeded:
                    raise
                except anthropic.RateLimitError:
                    wait = 2 ** (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                except Exception as e:
                    print(f"    Normalization error (attempt {attempt + 1}/3): {e}")
                    if attempt == 2:
                        # Fall back: map each skill to itself
                        for skill in chunk:
                            mapping[skill] = skill

        return mapping

    def _normalize_batch(self, chunks: List[List[str]]) -> Dict[str, str]:
        """Normalize skill chunks using the Batch API."""
        requests = []
        for i, chunk in enumerate(chunks):
            prompt = NORMALIZATION_PROMPT.format(skills_list=json.dumps(chunk, indent=2))
            requests.append(
                {
                    "custom_id": f"norm-{i}",
                    "params": {
                        "model": self.model,
                        "max_tokens": 4000,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
            )

        batch = self.client.messages.batches.create(requests=requests)
        print(f"  Normalization batch submitted: {batch.id} ({len(requests)} chunks)")

        batch_file = os.path.join(self.output_dir, "batch_ids.txt")
        with open(batch_file, "a") as f:
            f.write(f"{batch.id}\n")

        # Poll
        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            counts = batch.request_counts
            print(
                f"    Processing: {counts.processing} | "
                f"Succeeded: {counts.succeeded} | "
                f"Errored: {counts.errored}"
            )
            if batch.processing_status == "ended":
                break
            time.sleep(30)

        # Retrieve
        mapping: Dict[str, str] = {}
        for entry in self.client.messages.batches.results(batch.id):
            if entry.result.type == "succeeded":
                if self.cost_tracker:
                    usage = entry.result.message.usage
                    self.cost_tracker.update_from_usage(
                        usage.input_tokens, usage.output_tokens
                    )
                try:
                    raw_text = _strip_markdown_fences(entry.result.message.content[0].text)
                    chunk_mapping = json.loads(raw_text)
                    mapping.update(chunk_mapping)
                except Exception as e:
                    print(f"    Parse error for {entry.custom_id}: {e}")
            else:
                print(f"    Failed: {entry.custom_id} — {entry.result.type}")

        return mapping

    def apply_normalization(
        self, df: pd.DataFrame, use_batch: bool = True
    ) -> pd.DataFrame:
        """Collect unique skills, normalize them, and apply the mapping to the DataFrame."""
        categories = self.collect_unique_skills(df)
        all_skills = set()
        for skills in categories.values():
            all_skills.update(skills)

        print(f"\nNormalizing {len(all_skills)} unique skills...")
        if not all_skills:
            return df

        mapping = self.normalize_skills(sorted(all_skills), use_batch=use_batch)
        print(f"  Mapped {len(mapping)} skills to {len(set(mapping.values()))} canonical names")

        # Save mapping
        mapping_path = os.path.join(self.output_dir, "skill_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"  Saved mapping to {mapping_path}")

        # Apply mapping to DataFrame
        df = df.copy()
        for col in ["technical_skills", "soft_skills", "certifications", "domain_knowledge"]:
            if col not in df.columns:
                continue
            df[col] = df[col].apply(lambda val: self._apply_mapping(val, mapping))

        return df

    def _apply_mapping(self, val, mapping: Dict[str, str]) -> str:
        """Apply the normalization mapping to a JSON-encoded skill list."""
        if pd.isna(val):
            return val
        try:
            skills = json.loads(val) if isinstance(val, str) else val
            if not isinstance(skills, list):
                return val
            normalized = [mapping.get(s, s) for s in skills]
            return json.dumps(normalized)
        except (json.JSONDecodeError, TypeError):
            return val

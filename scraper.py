"""JobSpy wrapper with region support for UpskillScan."""

import time
from typing import List

import pandas as pd
from jobspy import scrape_jobs

from config import REGIONS, SCRAPE_HOURS_OLD


class JobScraper:
    """Scrapes job postings across US regions using python-jobspy."""

    def scrape_region(
        self, region_name: str, titles: List[str], n_per_city: int
    ) -> pd.DataFrame:
        """Scrape all cities in a single region for the given job titles."""
        if region_name not in REGIONS:
            raise ValueError(
                f"Unknown region '{region_name}'. Valid: {list(REGIONS.keys())}"
            )

        cities = REGIONS[region_name]
        all_dfs = []

        for city in cities:
            for title in titles:
                print(f"  Scraping '{title}' in {city}...")
                try:
                    df = scrape_jobs(
                        site_name=["indeed", "linkedin"],
                        search_term=title,
                        location=city,
                        results_wanted=n_per_city,
                        hours_old=SCRAPE_HOURS_OLD,
                        country_indeed="USA",
                    )
                    if df is not None and len(df) > 0:
                        df["region"] = region_name
                        df["city"] = city
                        df["search_title"] = title
                        all_dfs.append(df)
                        print(f"    Found {len(df)} postings")
                    else:
                        print(f"    No results")
                except Exception as e:
                    print(f"    WARNING: Failed to scrape '{title}' in {city}: {e}")

                time.sleep(2)  # Be polite to servers

        if not all_dfs:
            print(f"  No postings found for region {region_name}")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        return self._clean(combined)

    def scrape_all_regions(
        self,
        titles: List[str],
        n_per_city: int,
        region_names: List[str] = None,
    ) -> pd.DataFrame:
        """Scrape multiple regions. If region_names is None, scrape all."""
        regions_to_scrape = region_names or list(REGIONS.keys())
        all_dfs = []

        for region in regions_to_scrape:
            print(f"\n{'='*50}")
            print(f"Region: {region}")
            print(f"{'='*50}")
            df = self.scrape_region(region, titles, n_per_city)
            if len(df) > 0:
                all_dfs.append(df)

        if not all_dfs:
            print("No postings found across any region.")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = self._clean(combined)
        print(f"\nTotal postings after dedup: {len(combined)}")
        return combined

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate and drop rows missing descriptions."""
        before = len(df)
        # Drop missing descriptions
        df = df.dropna(subset=["description"])
        # Deduplicate on title + company_name + location
        dedup_cols = ["title", "company_name", "location"]
        existing_cols = [c for c in dedup_cols if c in df.columns]
        if existing_cols:
            df = df.drop_duplicates(subset=existing_cols, keep="first")
        after = len(df)
        if before != after:
            print(f"  Cleaned: {before} -> {after} postings (removed {before - after} dupes/empty)")
        return df.reset_index(drop=True)

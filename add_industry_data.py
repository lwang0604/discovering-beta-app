"""Refresh GICS sector and sub-industry fields for the app's S&P 1500 universe."""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
SOURCES = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "S&P MidCap 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "S&P SmallCap 600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def main() -> None:
    frames = []
    for source_index, url in SOURCES.items():
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
        table = table.rename(columns={"Symbol": "Ticker"})
        table["Ticker"] = table["Ticker"].str.replace(".", "-", regex=False)
        table["Source_Index"] = source_index
        frames.append(table[["Ticker", "GICS Sector", "GICS Sub-Industry", "Source_Index"]])

    industries = pd.concat(frames, ignore_index=True).drop_duplicates("Ticker")
    universe = pd.read_csv(ROOT / "universe.csv").drop(
        columns=["GICS Sector", "GICS Sub-Industry"], errors="ignore"
    )
    enriched = universe.merge(
        industries[["Ticker", "GICS Sector", "GICS Sub-Industry"]],
        on="Ticker",
        how="left",
    )
    enriched.to_csv(ROOT / "universe.csv", index=False)
    matched = enriched["GICS Sector"].notna().sum()
    print(f"Added industry data for {matched:,} of {len(enriched):,} companies.")


if __name__ == "__main__":
    main()

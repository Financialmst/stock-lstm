import requests
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
DATASET_PATH = "data/global_market_dataset.csv"
OUTPUT_PATH  = "data/global_market_dataset_with_short.csv"

def fetch_short_volume_for_date(date_str):
    """Fetch all short volume data for a single date and return as DataFrame."""
    url = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
    headers = {"Accept": "application/json"}
    params = {
        "limit": 9999,
        "filters": json.dumps([{
            "fieldName": "tradeReportDate",
            "fieldValue": date_str,
            "compareType": "EQUAL"
        }])
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df = df.rename(columns={
            "securitiesInformationProcessorSymbolIdentifier": "ticker",
            "tradeReportDate": "date",
            "shortParQuantity": "short_vol",
            "totalParQuantity": "total_vol",
        })
        # Filter to our tickers only
        df = df[df["ticker"].isin(TICKERS)]
        if df.empty:
            return pd.DataFrame()
        # Aggregate across market codes (NYSE + NASDAQ + BATS)
        df["short_vol"] = pd.to_numeric(df["short_vol"], errors="coerce").fillna(0)
        df["total_vol"] = pd.to_numeric(df["total_vol"], errors="coerce").fillna(0)
        agg = df.groupby(["date", "ticker"]).agg(
            short_vol=("short_vol", "sum"),
            total_vol=("total_vol", "sum"),
        ).reset_index()
        agg["short_vol_ratio"] = agg["short_vol"] / agg["total_vol"].replace(0, np.nan)
        return agg
    except Exception as e:
        print(f"  Error for {date_str}: {e}")
        return pd.DataFrame()


def build_date_range(start="2021-01-01", end="2026-06-01"):
    dates = []
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt  = datetime.strptime(end, "%Y-%m-%d")
    while current <= end_dt:
        if current.weekday() < 5:   # weekdays only
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def main():
    print("=" * 60)
    print("  FINRA Reg SHO Daily Short Volume Fetcher")
    print("=" * 60)

    dates = build_date_range()
    print(f"  Fetching {len(dates)} trading days for {TICKERS}")
    print("  This will take ~15-20 minutes (rate limited)\n")

    all_data = []
    for i, date_str in enumerate(dates):
        day_df = fetch_short_volume_for_date(date_str)
        if not day_df.empty:
            all_data.append(day_df)
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(dates)} days fetched...")
        time.sleep(0.4)   # ~2.5 req/sec — well within FINRA limits

    if not all_data:
        print("No data fetched.")
        return

    short_df = pd.concat(all_data, ignore_index=True)
    short_df["date"] = pd.to_datetime(short_df["date"])
    print(f"\n  Short volume data shape: {short_df.shape}")
    print(f"  Sample:\n{short_df.head()}")

    # Add rolling features
    print("\nBuilding rolling short volume features...")
    for ticker in TICKERS:
        mask = short_df["ticker"] == ticker
        short_df.loc[mask, "short_ratio_5d_avg"] = (
            short_df.loc[mask, "short_vol_ratio"].rolling(5).mean().values
        )
        short_df.loc[mask, "short_ratio_20d_avg"] = (
            short_df.loc[mask, "short_vol_ratio"].rolling(20).mean().values
        )
        short_df.loc[mask, "short_ratio_change"] = (
            short_df.loc[mask, "short_vol_ratio"].pct_change(5).values
        )

    # Save short data separately first
    short_df.to_csv("data/short_interest_raw.csv", index=False)
    print("  Saved raw short data → data/short_interest_raw.csv")

    # Merge into main dataset
    print("\nMerging into main dataset...")
    main_df = pd.read_csv(DATASET_PATH)
    main_df["Date"] = pd.to_datetime(main_df["Date"])

    merge_cols = ["date", "ticker", "short_vol_ratio",
                  "short_ratio_5d_avg", "short_ratio_20d_avg", "short_ratio_change"]
    short_merge = short_df[merge_cols].rename(columns={"date": "Date", "ticker": "Ticker"})

    merged = main_df.merge(short_merge, on=["Date", "Ticker"], how="left")
    merged[["short_vol_ratio", "short_ratio_5d_avg",
            "short_ratio_20d_avg", "short_ratio_change"]] = \
        merged[["short_vol_ratio", "short_ratio_5d_avg",
                "short_ratio_20d_avg", "short_ratio_change"]].fillna(0)

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved merged dataset → {OUTPUT_PATH}")
    print(f"  Shape: {merged.shape}")

    # Correlation diagnostic
    print("\n── Correlation Diagnostic ───────────────────────────────")
    short_cols = ["short_vol_ratio", "short_ratio_5d_avg",
                  "short_ratio_20d_avg", "short_ratio_change"]
    for ticker in TICKERS:
        t = merged[merged["Ticker"] == ticker].copy()
        if len(t) < 100:
            continue
        t["fwd_5d"] = t["Close_yfin"].shift(-5)
        t["direction"] = (t["fwd_5d"] > t["Close_yfin"] * 1.005).astype(int)
        t = t.dropna()
        print(f"  {ticker}:")
        for col in short_cols:
            corr = t[col].corr(t["direction"])
            print(f"    {col:30s}: {corr:+.4f}  {'✅ signal' if abs(corr) > 0.05 else '❌ noise'}")
        print()

    print("\nNext steps if correlation > 0.05:")
    print("  1. Update DATASET_PATH in train_transformer.py")
    print("     → 'data/global_market_dataset_with_short.csv'")
    print("  2. Add to FEATURE_COLUMNS:")
    print("     'short_vol_ratio', 'short_ratio_5d_avg',")
    print("     'short_ratio_20d_avg', 'short_ratio_change'")
    print("  3. python training/train_transformer.py --target direction")


if __name__ == "__main__":
    main()

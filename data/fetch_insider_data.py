import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

SEC_HEADERS = {
    "User-Agent": "FinIntelAI research@finintelai.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

TICKER_TO_CIK = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "NVDA":  "0001045810",
    "GOOGL": "0001652044",
    "AMZN":  "0001018724",
    "META":  "0001326801",
    "TSLA":  "0001318605",
}

DATASET_PATH = "data/global_market_dataset.csv"
OUTPUT_PATH  = "data/global_market_dataset_with_insider.csv"
WINDOW_DAYS  = [7, 30, 60]


def fetch_form4_dates(cik, ticker):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [{ticker}] Failed: {e}")
        return pd.Series(dtype="datetime64[ns]")

    filings = data.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    dates   = filings.get("filingDate", [])

    filing_dates = [
        pd.to_datetime(date)
        for form, date in zip(forms, dates)
        if form == "4"
    ]

    older = data.get("filings", {}).get("files", [])
    for f in older:
        try:
            old_url = f"https://data.sec.gov/submissions/{f['name']}"
            old_resp = requests.get(old_url, headers={
                "User-Agent": "FinIntelAI research@finintelai.com"
            }, timeout=15)
            old_data = old_resp.json()
            for form, date in zip(old_data.get("form", []), old_data.get("filingDate", [])):
                if form == "4":
                    filing_dates.append(pd.to_datetime(date))
            time.sleep(0.2)
        except Exception:
            pass

    print(f"  [{ticker}] {len(filing_dates)} Form 4 filings found")
    return pd.Series(filing_dates)


def build_insider_features(filing_dates, price_df):
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)

    for w in WINDOW_DAYS:
        price_df[f"insider_filings_{w}d"] = 0.0
    price_df["insider_activity_7d"] = 0.0

    if filing_dates.empty:
        return price_df

    for idx in price_df.index:
        for w in WINDOW_DAYS:
            cutoff = idx - timedelta(days=w)
            count = int(((filing_dates >= cutoff) & (filing_dates <= idx)).sum())
            price_df.at[idx, f"insider_filings_{w}d"] = count
        cutoff_7 = idx - timedelta(days=7)
        price_df.at[idx, "insider_activity_7d"] = int(
            ((filing_dates >= cutoff_7) & (filing_dates <= idx)).sum() > 0
        )

    return price_df


def run_correlation_diagnostic(df):
    print("\n── Correlation Diagnostic ───────────────────────────────")
    print("Insider features vs 5-day forward direction:\n")
    insider_cols = [c for c in df.columns if "insider" in c]

    for ticker in TICKER_TO_CIK.keys():
        t = df[df["Ticker"] == ticker].copy()
        if len(t) < 100:
            continue
        t["fwd_5d"] = t["Close_yfin"].shift(-5)
        t["direction"] = (t["fwd_5d"] > t["Close_yfin"] * 1.005).astype(int)
        t = t.dropna()
        print(f"  {ticker}:")
        for col in insider_cols:
            corr = t[col].corr(t["direction"])
            print(f"    {col:30s}: {corr:+.4f}  {'✅ signal' if abs(corr) > 0.05 else '❌ noise'}")
        print()


def main():
    print("=" * 60)
    print("  SEC EDGAR Form 4 Insider Data Fetcher")
    print("=" * 60)

    print(f"\nLoading {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Shape: {df.shape}")

    for w in WINDOW_DAYS:
        df[f"insider_filings_{w}d"] = 0.0
    df["insider_activity_7d"] = 0.0

    print("\nFetching SEC EDGAR Form 4 filings...")
    all_filings = {}
    for ticker, cik in TICKER_TO_CIK.items():
        all_filings[ticker] = fetch_form4_dates(cik, ticker)
        time.sleep(0.5)

    print("\nBuilding rolling insider features...")
    updated_dfs = []

    for ticker in df["Ticker"].unique():
        ticker_df = df[df["Ticker"] == ticker].copy()
        has_date_col = "Date" in ticker_df.columns
        if has_date_col:
            ticker_df = ticker_df.set_index("Date")

        if ticker in all_filings and not all_filings[ticker].empty:
            ticker_df = build_insider_features(all_filings[ticker], ticker_df)
            print(f"  {ticker}: features built ✅")
        else:
            print(f"  {ticker}: no SEC data, features set to 0")

        if has_date_col:
            ticker_df = ticker_df.reset_index()
        updated_dfs.append(ticker_df)

    df_updated = pd.concat(updated_dfs, ignore_index=True)
    run_correlation_diagnostic(df_updated)

    df_updated.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved: {OUTPUT_PATH}")
    print(f"   Shape: {df_updated.shape}")
    print(f"   New columns: {[c for c in df_updated.columns if 'insider' in c]}")
    print("\nNext steps:")
    print("  1. Check correlations above — any > 0.05 is worth retraining on")
    print("  2. Update DATASET_PATH in train_transformer.py to use the new file")
    print("  3. Add insider_filings_30d, insider_filings_60d, insider_activity_7d to FEATURE_COLUMNS")
    print("  4. python training/train_transformer.py --target direction")


if __name__ == "__main__":
    main()

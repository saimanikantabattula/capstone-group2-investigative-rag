"""
build_manifest.py

Reads the full 100k IRS structured index CSV and produces a balanced
25,000-row sample covering a mix of return types and filing years.
Output is saved as data/manifests/irs_manifest_25k.csv.
"""

import os
import pandas as pd
from pathlib import Path


IRS_CSV = "/Users/battulasaimanikanta/Documents/capstone data sets /dataset 1.0/structured/IRS990_structured_index_sample_2024_2025_100k.csv"
OUTPUT_DIR = "/Users/battulasaimanikanta/Documents/capstone-group2-investigative-rag/data/manifests"
OUTPUT_FILE = "irs_manifest_25k.csv"
TARGET = 25000


def load_csv(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def clean(df):
    required = ["OBJECT_ID", "XML_BATCH_ID"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")
    df = df.dropna(subset=required)
    df["OBJECT_ID"] = df["OBJECT_ID"].str.strip()
    df["XML_BATCH_ID"] = df["XML_BATCH_ID"].str.strip()
    df = df[df["OBJECT_ID"] != ""]
    df = df[df["XML_BATCH_ID"] != ""]
    return df.drop_duplicates(subset=["OBJECT_ID"])


def sample_balanced(df, target):
    # sample proportionally across return types so we get coverage of
    # 990, 990EZ, 990T, 990PF etc. rather than just the most common type
    if "RETURN_TYPE" not in df.columns:
        return df.sample(n=min(target, len(df)), random_state=42)

    groups = df.groupby("RETURN_TYPE")
    n_groups = groups.ngroups
    per_group = target // n_groups

    parts = []
    remainder = target

    for return_type, group in groups:
        n = min(per_group, len(group))
        parts.append(group.sample(n=n, random_state=42))
        remainder -= n

    combined = pd.concat(parts, ignore_index=True)

    # fill remaining slots from rows not yet selected
    if remainder > 0:
        selected_ids = set(combined["OBJECT_ID"])
        leftover = df[~df["OBJECT_ID"].isin(selected_ids)]
        if len(leftover) > 0:
            extra = leftover.sample(n=min(remainder, len(leftover)), random_state=42)
            combined = pd.concat([combined, extra], ignore_index=True)

    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    print("Loading IRS CSV...")
    df = load_csv(IRS_CSV)
    print(f"Loaded {len(df)} rows")

    df = clean(df)
    print(f"After cleaning: {len(df)} valid rows")

    if "RETURN_TYPE" in df.columns:
        print("Return type breakdown:")
        print(df["RETURN_TYPE"].value_counts().to_string())

    sample = sample_balanced(df, TARGET)
    print(f"\nSampled {len(sample)} rows")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    out = output_path / OUTPUT_FILE
    sample.to_csv(out, index=False)
    print(f"Saved to: {out}")

    if "RETURN_TYPE" in sample.columns:
        print("\nSample return type distribution:")
        print(sample["RETURN_TYPE"].value_counts().to_string())


if __name__ == "__main__":
    main()

import argparse
import glob
import os
import pandas as pd
from recommender import RecommendationEngine

def find_first_csv_in_data(default_path="data"):
    candidates = glob.glob(os.path.join(default_path, "*.csv"))
    return candidates[0] if candidates else None

def main():
    parser = argparse.ArgumentParser(description="Build similarity index for music recommender")
    parser.add_argument("--csv", type=str, default=None, help="Path to dataset CSV (default: first CSV in ./data)")
    parser.add_argument("--sample", type=int, default=None, help="Random sample size for speed (optional)")
    parser.add_argument("--topk", type=int, default=50, help="Neighbors to index for diagnostics (does not affect runtime quality)")
    args = parser.parse_args()

    csv_path = args.csv or find_first_csv_in_data()
    if not csv_path or not os.path.exists(csv_path):
        raise SystemExit("No CSV found. Put a file in ./data or pass --csv path.")

    print(f"[INFO] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"[INFO] Downsampled to {len(df)} rows for speed.")

    engine = RecommendationEngine()
    X = engine.fit(df)

    os.makedirs("models", exist_ok=True)
    engine.save("models/recommender.joblib", "models/feature_meta.json")
    print("[OK] Saved models to models/recommender.joblib and models/feature_meta.json")
    print(f"[INFO] Indexed rows: {len(df)}")

if __name__ == "__main__":
    main()

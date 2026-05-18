"""Build universal risk_features parquet (8-d day-memory key inputs).

Same shape as ``data/processed/risk_features.parquet`` but with the
three biotech-tied slots populated from XLK instead of XBI:

    xbi_rv_20d           <- XLK realised vol 20d  (column name kept)
    xbi_rv_60d           <- XLK realised vol 60d
    xbi_fwd_abs_ret_5d   <- |XLK 5d log return|

Output: data/processed/risk_features_sp500.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    base = pd.read_parquet("data/processed/risk_features.parquet")
    base.index = pd.to_datetime(base.index)
    print(f"base risk_features: shape={base.shape} cols={base.columns.tolist()}",
          flush=True)

    etfs = pd.read_parquet("data/raw/sp500/sector_etfs.parquet")
    etfs["date"] = pd.to_datetime(etfs["date"]).dt.normalize()
    xlk = etfs[etfs.ticker == "XLK"].set_index("date")["close"].sort_index()
    xlk_aligned = xlk.reindex(base.index).ffill(limit=5)

    logret_1d = np.log(xlk_aligned / xlk_aligned.shift(1))
    out = base.copy()
    # XLK realised vols replace XBI
    out["xbi_rv_20d"] = logret_1d.rolling(20).std()
    out["xbi_rv_60d"] = logret_1d.rolling(60).std()
    # XLK forward 5d absolute return replaces XBI
    fwd_5d = np.log(xlk_aligned.shift(-5) / xlk_aligned).abs()
    out["xbi_fwd_abs_ret_5d"] = fwd_5d

    out_path = Path("data/processed/risk_features_sp500.parquet")
    out.to_parquet(out_path)
    print(f"wrote {out_path}: shape={out.shape}", flush=True)
    print(f"  xbi_rv_20d (now XLK): mean={out.xbi_rv_20d.mean():.5f} "
          f"std={out.xbi_rv_20d.std():.5f}", flush=True)
    print(f"  xbi_rv_60d (now XLK): mean={out.xbi_rv_60d.mean():.5f} "
          f"std={out.xbi_rv_60d.std():.5f}", flush=True)


if __name__ == "__main__":
    main()

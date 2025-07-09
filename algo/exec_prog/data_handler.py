# algo/data_handler.py

from pathlib import Path
import pandas as pd
from features import add_indicators

class DataHandler:
    def __init__(
        self,
        raw_dir: str      = "data",
        proc_dir: str     = "data_processed",
        suffix_raw: str   = ".csv",
        suffix_proc: str  = "_feat.csv"
    ):
        self.raw_dir   = Path(raw_dir)
        self.proc_dir  = Path(proc_dir)
        self.suffix_raw  = suffix_raw
        self.suffix_proc = suffix_proc
        self.proc_dir.mkdir(exist_ok=True)

    def _clean_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip whitespace/newlines from column names
        df.columns = df.columns.str.strip()
        return df

    def load_raw(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Load the raw OHLCV bars from data/<symbol>_<interval>.csv
        """
        path = self.raw_dir / f"{symbol}_{interval}{self.suffix_raw}"
        df   = pd.read_csv(path, index_col=0, parse_dates=True)
        return self._clean_cols(df)

    def load_features(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Incrementally build or update the feature-augmented CSV.
        - If no cache exists, builds from scratch.
        - If cache exists, reads it, finds new raw rows, computes features, appends them.
        """
        raw_path = self.raw_dir / f"{symbol}_{interval}{self.suffix_raw}"
        proc_path = self.proc_dir / f"{symbol}_{interval}{self.suffix_proc}"

        # 1) Load raw + clean
        df_raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        df_raw = self._clean_cols(df_raw)

        # 2) If no processed cache, build full
        if not proc_path.exists():
            df_full = add_indicators(df_raw).ffill()
            df_full.to_csv(proc_path)
            return df_full

        # 3) Otherwise load existing features
        df_proc = pd.read_csv(proc_path, index_col=0, parse_dates=True)
        df_proc = self._clean_cols(df_proc)

        # 4) Identify new timestamps in raw
        new_idx = df_raw.index.difference(df_proc.index)
        if new_idx.empty:
            return df_proc  # nothing to do

        # 5) Feature-engineer only the new slice
        df_new_raw = df_raw.loc[new_idx]
        df_new_feat = add_indicators(df_new_raw).ffill()

        # 6) Append & re-sort
        df_updated = pd.concat([df_proc, df_new_feat]).sort_index()

        # 7) Overwrite cache
        df_updated.to_csv(proc_path)
        return df_updated


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    # ── Interactive prompts ────────────────────────────────────────────
    symbol   = input("Enter symbol (e.g. RELIANCE): ").strip().upper()
    interval = input("Enter interval (e.g. 5minute): ").strip()
    raw_dir  = input("Raw data folder (default 'data'): ").strip() or "data"
    proc_dir = input("Processed folder (default 'data_processed'): ").strip() or "data_processed"

    # ── Build / load feature DataFrame ──────────────────────────────────
    dh = DataHandler(raw_dir=raw_dir, proc_dir=proc_dir)
    df = dh.load_features(symbol=symbol, interval=interval)

    # ── Summarize results ───────────────────────────────────────────────
    out_path = Path(proc_dir) / f"{symbol}_{interval}_feat.csv"
    print(f"\n✅ Processed file: {out_path}")
    print("Exists on disk:", out_path.exists())
    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist(), "\n")
    print("First 5 rows:")
    print(df.head(5).to_string())
    print("\nLast 5 rows:")
    print(df.tail(5).to_string())

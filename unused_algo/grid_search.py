import warnings
# Suppress pandas_ta warnings globally
warnings.filterwarnings('ignore', category=UserWarning, module='pandas_ta')

import os
import pathlib
import pandas as pd
import numpy as np
import itertools
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from algo.config import load_config
from algo.broker import KiteWrapper
from algo.features import add_indicators
from algo.reg_model import load_or_train_reg, predict_last_reg
from algo.backtester import backtest

def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 6
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    total = len(df)
    fold_size = total // (n_splits + 1)
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end  = fold_size * (i + 2)
        train_df  = df.iloc[:train_end].copy()
        test_df   = df.iloc[train_end:test_end].copy()
        if test_df.empty:
            break
        splits.append((train_df, test_df))
    return splits


def evaluate_params_task(args):
    warnings.filterwarnings('ignore', category=UserWarning, module='pandas_ta')
    combo, raw_df, folds, keys = args
    hyperparams = dict(zip(keys, combo))
    pnl_list: List[float] = []
    for train_df, test_df in folds:
        df_train = add_indicators(train_df).ffill().dropna()
        df_test  = add_indicators(test_df).ffill().dropna()
        model = load_or_train_reg(df_train, retrain=True)
        _, metrics = backtest(
            df_raw        = df_test,
            model         = model,
            predict_fn    = predict_last_reg,
            skip_indicator= True,
            **hyperparams
        )
        pnl_list.append(metrics['PnL'])
    mean_pnl = float(np.mean(pnl_list))
    std_pnl  = float(np.std(pnl_list))
    # Always return summary, including negative PnL, so you can inspect results
    return {**hyperparams, 'pnl_folds': pnl_list, 'pnl_mean': mean_pnl, 'pnl_std': std_pnl}



def run_grid_search():
    # Determine project root and results file
    proj_root = pathlib.Path(__file__).parent.parent.resolve()
    results_file = proj_root / "grid_search_robust.csv"
    print(f"Project root: {proj_root}")
    print(f"Results CSV: {results_file}")

    cfg    = load_config()
    broker = KiteWrapper(cfg)
    raw = broker.history(days=180, interval='3minute', tradingsymbol='HDFCBANK')
    folds = walk_forward_splits(raw, n_splits=6)

    param_grid = {
        'capital':      [100000],
        'contract_size':[1],
        'sl_pct':       np.linspace(0.0002, 0.002, 5),
        'tp_pct':       np.linspace(0.001, 0.005, 5),
        'trail_pct':    np.linspace(0.0001, 0.001, 5),
        'hold_max':     [3,5,8],
        'upper':        np.linspace(0.5, 0.9, 5),
        'lower':        np.linspace(0.1, 0.4, 5),
    }
    combos = list(itertools.product(*param_grid.values()))
    keys   = list(param_grid.keys())

    # Initialize or resume results
    if results_file.exists():
        done_df = pd.read_csv(results_file)
        done_set = {tuple(row[k] for k in keys) for _, row in done_df.iterrows()}
        results = done_df.to_dict('records')
    else:
        # Create empty CSV with headers so you can open it immediately
        header_cols = keys + ['pnl_folds', 'pnl_mean', 'pnl_std']
        pd.DataFrame(columns=header_cols).to_csv(results_file, index=False)
        print(f"Initialized results file with headers: {results_file}")
        # Show directory listing
        print("Directory contents:", sorted(os.listdir(proj_root)))
        done_set = set()
        results = []

    tasks = [(combo, raw, folds, keys) for combo in combos if combo not in done_set]

    # Parallel execution with progress bar
    n_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_params_task, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Grid Search', unit='param'):
            combo = futures[future][0]
            try:
                res = future.result()
                if res:
                    results.append(res)
                    # Append new result without writing header again
                    pd.DataFrame([res]).to_csv(
                        results_file,
                        mode='a',
                        header=False,
                        index=False
                    )
            except Exception:
                continue

    print(f"Total robust sets found: {len(results)}")

if __name__ == '__main__':
    run_grid_search()

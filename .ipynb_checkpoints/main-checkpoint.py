"""Entry script – keeps top level crisp so you can unit‑test package parts."""

import argparse
from algo.runner import run_live

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Zerodha ML scalper (algo package)")
    p.add_argument("--symbol", default="BANKNIFTY24JUNFUT")
    p.add_argument("--qty", type=int, default=15)
    p.add_argument("--live", action="store_true", help="Actually place real orders")
    args = p.parse_args()

    run_live(symbol=args.symbol, qty=args.qty, live=args.live)
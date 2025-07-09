# paper_trade.py
from algo import run_live

if __name__ == "__main__":
    # This is now the real “main” interpreter,
    # so KiteTicker’s reactor can install signals safely.
    run_live(symbol="RELIANCE", qty=1, live=False)
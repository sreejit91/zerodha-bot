import csv, os
from datetime import datetime, timezone, timedelta
from algo.broker import _resolve_token

# IST timezone object
IST = timezone(timedelta(hours=5, minutes=30))

class TickAggregator:
    """
    Aggregates raw FULL-mode ticks into OHLCV bars AND computes:
      • buy_volume   = end_buy - start_buy
      • sell_volume  = end_sell - start_sell
      • imbalance    = (buy_volume - sell_volume) / (buy_volume + sell_volume)
    """

    def __init__(self, kite_wrapper, symbol: str, intervals: list[int], data_dir: str):
        self.kw        = kite_wrapper
        self.symbol    = symbol
        self.intervals = intervals
        self.data_dir  = data_dir

        # resolve & store instrument token
        self.token = _resolve_token(self.kw.kite, self.symbol, self.kw.cfg.exchange)

        # Per-interval state
        self.states = {}
        for sec in intervals:
            self.states[sec] = {
                "open": None, "high": None, "low": None, "close": None,
                "vol": 0,
                "start_buy": None, "start_sell": None,
                "end_buy": None,   "end_sell": None,
                "last_ts": None
            }

            os.makedirs(data_dir, exist_ok=True)
            fn = f"{self.symbol}_{sec}s.csv"
            path = os.path.join(data_dir, fn)
            if not os.path.exists(path):
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "timestamp","open","high","low","close","volume",
                        "buy_volume","sell_volume","imbalance"
                    ])

    def on_tick(self, tick: dict):
        # pick the exchange timestamp (string) and parse as IST-aware
        ts_str = tick.get("last_trade_time") or tick.get("exchange_timestamp")
        if isinstance(ts_str, str):
            ts = datetime.fromisoformat(ts_str)
        else:
            ts = ts_str
        # assume incoming timestamps are in IST
        ts = ts.replace(tzinfo=IST)

        # round down to the current minute
        ts = ts.replace(second=0, microsecond=0)

        for sec, st in self.states.items():
            # compute which bar this tick belongs to
            bar_min = (ts.minute // (sec // 60)) * (sec // 60)
            bar_ts = ts.replace(minute=bar_min)

            # new bar?
            if st["last_ts"] != bar_ts:
                # flush previous bar
                if st["last_ts"] is not None:
                    self._close_bar(sec, st)
                # start new
                price = tick["last_price"]
                st.update({
                    "open": price, "high": price, "low": price, "close": price,
                    "vol": 0,
                    "start_buy": tick["total_buy_quantity"],
                    "start_sell": tick["total_sell_quantity"],
                    "end_buy": tick["total_buy_quantity"],
                    "end_sell": tick["total_sell_quantity"],
                    "last_ts": bar_ts
                })
            else:
                price = tick["last_price"]
                st["high"]  = max(st["high"], price)
                st["low"]   = min(st["low"], price)
                st["close"] = price
                st["vol"]  += tick.get("last_traded_quantity", 0)
                st["end_buy"]  = tick["total_buy_quantity"]
                st["end_sell"] = tick["total_sell_quantity"]

    def _close_bar(self, sec: int, st: dict):
        # write out CSV row
        ts_iso = st["last_ts"].isoformat()
        buy_vol  = (st["end_buy"]  - st["start_buy"])  if st["start_buy"] is not None else 0
        sell_vol = (st["end_sell"] - st["start_sell"]) if st["start_sell"] is not None else 0
        imbalance = 0.0
        if buy_vol + sell_vol > 0:
            imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)

        row = [
            ts_iso,
            st["open"], st["high"], st["low"], st["close"],
            st["vol"],
            buy_vol, sell_vol, imbalance
        ]

        path = os.path.join(self.data_dir, f"{self.symbol}_{sec}s.csv")
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

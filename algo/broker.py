"""
algo/broker.py
Zerodha wrapper that auto-resolves instrument tokens and caches them.
Timestamps are converted to Asia/Kolkata so backtests align with local market hours.
Prevents infinite loops by ensuring the cursor always advances.
"""

import datetime as _dt
import time
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from kiteconnect import KiteConnect, KiteTicker
from kiteconnect.exceptions import DataException

from .config import load_config
from .logger import Order

print(f"[broker] loaded from {__file__}")

@lru_cache(maxsize=None)
def _resolve_token(kite: KiteConnect, tradingsymbol: str, exchange: str) -> int:
    """
    Return the instrument token for a given tradingsymbol on the given exchange.
    """
    for inst in kite.instruments(exchange):
        if inst["tradingsymbol"] == tradingsymbol:
            return inst["instrument_token"]
    raise ValueError(f"{tradingsymbol!r} not found on {exchange}")

class KiteWrapper:
    def __init__(self, cfg: Optional[Any] = None):
        self.cfg = cfg or load_config()
        self.kite = KiteConnect(api_key=self.cfg.api_key)
        self.kite.set_access_token(self.cfg.access_token)
        self._ticker: Optional[KiteTicker] = None
        print(f"[KiteWrapper] initialized: symbol={self.cfg.tradingsymbol} on exch={self.cfg.exchange}")

    def history(
        self,
        *,
        days: int | None = None,
        from_date: _dt.datetime | str | None = None,
        to_date: _dt.datetime | str | None = None,
        interval: str = "5minute",
        tradingsymbol: str | None = None,
        continuous: bool = False,
        oi: bool = False,
    ) -> pd.DataFrame:
        print(f"[history] start: days={days}, interval={interval}, symbol={tradingsymbol}", flush=True)

        # Determine date range
        if days is None and (from_date is None or to_date is None):
            raise ValueError("Provide days=N or from_date/to_date.")
        if from_date is None or to_date is None:
            to_date = _dt.datetime.utcnow()
            from_date = to_date - _dt.timedelta(days=days)
        else:
            if isinstance(from_date, str): from_date = _dt.datetime.fromisoformat(from_date)
            if isinstance(to_date, str):   to_date   = _dt.datetime.fromisoformat(to_date)
        # Normalize to UTC naive
        if from_date.tzinfo:
            from_date = from_date.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        if to_date.tzinfo:
            to_date   = to_date.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        print(f"[history] range UTC-naive: {from_date} → {to_date}", flush=True)

        # Chunk size based on interval
        intr1 = {"minute"}
        intr5 = {"3minute","5minute","10minute","15minute","30minute"}
        hr    = {"60minute","hour"}
        if interval in intr1:
            span_max = 60
        elif interval in intr5:
            span_max = 100
        elif interval in hr or "hour" in interval:
            span_max = 247
        else:
            span_max = 10_000

        # Resolve token
        symbol = tradingsymbol or self.cfg.tradingsymbol
        token  = _resolve_token(self.kite, symbol, self.cfg.exchange)
        print(f"[history] token={token}", flush=True)

        dfs         = []
        cursor      = from_date
        prev_cursor = None

        # Fetch loop
        while cursor < to_date:
            # Break if cursor didn't advance
            if prev_cursor is not None and cursor <= prev_cursor:
                print(f"[history] cursor stuck at {cursor}, breaking loop", flush=True)
                break
            prev_cursor = cursor

            fetched = False
            for att in range(5):
                span = max(1, span_max // (2**att))
                end  = min(cursor + _dt.timedelta(days=span), to_date)
                try:
                    raw = self.kite.historical_data(
                        instrument_token=token,
                        from_date=cursor.strftime("%Y-%m-%d %H:%M:%S"),
                        to_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                        interval=interval,
                        continuous=continuous,
                        oi=oi,
                    )
                    dfc = pd.DataFrame(raw)

                    # Empty response -> no more data
                    if dfc.empty:
                        print(f"[history] empty data for {cursor}->{end}, breaking loop", flush=True)
                        cursor = to_date
                        break

                    # Parse and localize to IST
                    dfc['date'] = pd.to_datetime(dfc['date'], utc=True)
                    dfc['date'] = dfc['date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                    last = dfc['date'].max().to_pydatetime().replace(tzinfo=None)

                    # Append and advance
                    dfs.append(dfc)
                    if 'minute' in interval:
                        mins = int(''.join(filter(str.isdigit, interval)))
                        adv  = _dt.timedelta(minutes=mins)
                    elif 'hour' in interval:
                        hrs = int(''.join(filter(str.isdigit, interval)))
                        adv = _dt.timedelta(hours=hrs)
                    else:
                        adv = _dt.timedelta(seconds=0)
                    cursor = last + adv

                    print(f"[history] got {len(dfc)} bars, cursor→{cursor}", flush=True)
                    fetched = True
                    break
                except DataException as e:
                    wait = 2**att
                    print(f"503 err {cursor}->{end}: {e}, retry {wait}s", flush=True)
                    time.sleep(wait)
            if not fetched:
                break

        if not dfs:
            raise RuntimeError("No data fetched.")

        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates('date').set_index('date').sort_index()
        # Filter only regular market hours (09:15 to 15:30 IST)
        df = df.between_time('09:15', '15:30')
        print(f"[history] complete {len(df)} bars {df.index.min()} → {df.index.max()}", flush=True)
        return df

    @property
    def ticker(self) -> KiteTicker:
        if self._ticker is None:
            self._ticker = KiteTicker(api_key=self.cfg.api_key, access_token=self.cfg.access_token)
        return self._ticker

    def place_order(
        self, tradingsymbol: str, quantity: int, transaction_type: str, live: bool = False
    ) -> Order:
        key = f"{self.cfg.exchange}:{tradingsymbol}"
        ltp = self.kite.ltp([key])[key]['last_price']
        oid = None
        if live:
            oid = self.kite.place_order(
                variety='regular', exchange=self.cfg.exchange,
                tradingsymbol=tradingsymbol, transaction_type=transaction_type,
                quantity=quantity, product='MIS', order_type='MARKET'
            )
        return Order(timestamp=_dt.datetime.now(_dt.timezone.utc),
                     symbol=tradingsymbol, side=transaction_type,
                     qty=quantity, price=ltp, broker_id=oid)

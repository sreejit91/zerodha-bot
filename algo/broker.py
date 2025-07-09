# algo/broker.py

import datetime as _dt
import time
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from kiteconnect import KiteConnect, KiteTicker
from kiteconnect.exceptions import DataException

from algo.config import load_config
from algo.logger import Order

# You'll need pytz for explicit timezone conversions:
import pytz

print(f"[broker] loaded from {__file__}")

@lru_cache(maxsize=None)
def _resolve_token(kite: KiteConnect, tradingsymbol: str, exchange: str) -> int:
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

        # -- determine raw from/to in UTC naive
        utc = pytz.UTC
        ist = pytz.timezone("Asia/Kolkata")

        if from_date is None or to_date is None:
            to_utc = _dt.datetime.utcnow().replace(tzinfo=utc)
            from_utc = to_utc - _dt.timedelta(days=days)
        else:
            # parse strings if needed
            if isinstance(to_date, str):
                to_utc = utc.localize(_dt.datetime.fromisoformat(to_date))
            else:
                to_utc = to_date.astimezone(utc)
            if isinstance(from_date, str):
                from_utc = utc.localize(_dt.datetime.fromisoformat(from_date))
            else:
                from_utc = from_date.astimezone(utc)

        # -- reset the 'from' to that date at 09:15 IST
        from_ist = from_utc.astimezone(ist)
        from_ist = from_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        from_utc = from_ist.astimezone(utc)

        # strip tzinfo for Kite API
        from_naive = from_utc.replace(tzinfo=None)
        to_naive   = to_utc.replace(tzinfo=None)
        print(f"[history] range UTC-naive: {from_naive} → {to_naive}", flush=True)

        # decide chunk size as before...
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

        # resolve token
        symbol = tradingsymbol or self.cfg.tradingsymbol
        token  = _resolve_token(self.kite, symbol, self.cfg.exchange)
        print(f"[history] token={token}", flush=True)

        dfs = []
        cursor = from_naive
        prev_cursor = None

        # fetch in chunks
        while cursor < to_naive:
            if prev_cursor is not None and cursor <= prev_cursor:
                print(f"[history] cursor stuck at {cursor}, breaking loop", flush=True)
                break
            prev_cursor = cursor

            fetched = False
            for att in range(5):
                span = max(1, span_max // (2**att))
                end = min(cursor + _dt.timedelta(days=span), to_naive)
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
                    if dfc.empty:
                        cursor = to_naive
                        break

                    # parse, convert to IST, drop tz
                    dfc["date"] = pd.to_datetime(dfc["date"], utc=True)
                    dfc["date"] = dfc["date"].dt.tz_convert(ist).dt.tz_localize(None)
                    last = dfc["date"].max().to_pydatetime()

                    dfs.append(dfc)

                    # advance cursor by one bar
                    if interval == "minute":
                        adv = _dt.timedelta(minutes=1)
                    elif interval == "hour":
                        adv = _dt.timedelta(hours=1)
                    elif interval.endswith("minute"):
                        adv = _dt.timedelta(minutes=int(interval.rstrip("minute")))
                    elif interval.endswith("hour"):
                        adv = _dt.timedelta(hours=int(interval.rstrip("hour")))
                    else:
                        adv = _dt.timedelta(0)

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
        df = df.drop_duplicates("date").set_index("date").sort_index()

        # keep only regular market hours
        df = df.between_time("09:15", "15:30")
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
        ltp = self.kite.ltp([key])[key]["last_price"]
        oid = None
        if live:
            oid = self.kite.place_order(
                variety="regular",
                exchange=self.cfg.exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product="MIS",
                order_type="MARKET",
            )
        return Order(
            timestamp=_dt.datetime.now(_dt.timezone.utc),
            symbol=tradingsymbol,
            side=transaction_type,
            qty=quantity,
            price=ltp,
            broker_id=oid,
        )

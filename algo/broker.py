"""
algo/broker.py
Zerodha wrapper that auto-resolves instrument tokens and caches them.
"""

from typing import Any, Optional
from functools import lru_cache
import datetime as _dt

import pandas as pd
from kiteconnect import KiteConnect, KiteTicker

from .config import load_config
from .logger import Order


# ────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _resolve_token(kite: KiteConnect, tradingsymbol: str, exchange: str) -> int:
    """
    Scan the master instrument file once per session and return the
    current token for (exchange, tradingsymbol).  Cached via lru_cache.
    """
    for inst in kite.instruments(exchange):
        if inst["tradingsymbol"] == tradingsymbol:
            return inst["instrument_token"]
    raise ValueError(f"{tradingsymbol!r} not found on {exchange}")
# ────────────────────────────────────────────────────────────────────


class KiteWrapper:
    """Facade hiding kiteconnect boilerplate + token refresh."""

    def __init__(self, cfg: Optional[Any] = None):
        self.cfg = cfg or load_config()

        self.kite = KiteConnect(api_key=self.cfg.api_key)
        self.kite.set_access_token(self.cfg.access_token)

        self._ticker: Optional[KiteTicker] = None

    # ─────────────────────────────────────────────────────────────
    # Historical data
    # ─────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────
    # Historical data – auto-chunked (no 100-day / 247-day limits)
    # ─────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────
    # Historical data – auto-chunked (handles all Kite limits)
    # ─────────────────────────────────────────────────────────────
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
        """
        Fetch candles of any span/interval without hitting Kite's
        per-request limits (60 d for 1-min, 100 d for ≤30-min, 247 d for 60-min).
        Either give *days=N* or *from_date / to_date*.
        """
        # -------- resolve date span --------
        if days is None and (from_date is None or to_date is None):
            raise ValueError("Provide days=N or from_date/to_date.")

        if from_date is None or to_date is None:
            to_date = _dt.datetime.utcnow()
            from_date = to_date - _dt.timedelta(days=days)
        else:
            if isinstance(from_date, str):
                from_date = _dt.datetime.fromisoformat(from_date)
            if isinstance(to_date, str):
                to_date = _dt.datetime.fromisoformat(to_date)

        # -------- per-request day cap --------
        intraday_1m = {"minute"}
        intraday_5m = {"3minute", "5minute", "10minute",
                       "15minute", "30minute"}
        hourly = {"60minute", "hour"}

        if interval in intraday_1m:
            span_max = 60
        elif interval in intraday_5m:
            span_max = 100
        elif interval in hourly or "hour" in interval:
            span_max = 247
        else:  # day / week / month
            span_max = 10_000

        # -------- token --------
        sym = tradingsymbol or self.cfg.tradingsymbol
        token = _resolve_token(self.kite, sym, self.cfg.exchange)

        # -------- chunked download --------
        dfs: list[pd.DataFrame] = []
        cursor = from_date
        while cursor < to_date:
            chunk_end = min(cursor + _dt.timedelta(days=span_max), to_date)
            raw = self.kite.historical_data(
                instrument_token=token,
                from_date=cursor,
                to_date=chunk_end,
                interval=interval,
                continuous=continuous,
                oi=oi,
            )
            dfs.append(pd.DataFrame(raw))
            cursor = chunk_end

        if not dfs:
            raise RuntimeError("No data returned")

        df = (pd.concat(dfs, ignore_index=True)
              .drop_duplicates("date")
              .set_index("date")
              .sort_index())

        return df

    # ─────────────────────────────────────────────────────────────
    # Lazy websocket
    # ─────────────────────────────────────────────────────────────
    @property
    def ticker(self) -> KiteTicker:
        if self._ticker is None:
            self._ticker = KiteTicker(
                api_key=self.cfg.api_key,
                access_token=self.cfg.access_token,
            )
        return self._ticker

    # ─────────────────────────────────────────────────────────────
    # Live / paper order helper
    # ─────────────────────────────────────────────────────────────
    def place_order(
        self,
        tradingsymbol: str,
        quantity: int,
        transaction_type: str,   # "BUY" | "SELL"
        live: bool = False,
    ) -> Order:
        ltp_key = f"{self.cfg.exchange}:{tradingsymbol}"
        ltp = self.kite.ltp([ltp_key])[ltp_key]["last_price"]

        order_id = None
        if live:
            order_id = self.kite.place_order(
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
            broker_id=order_id,
        )

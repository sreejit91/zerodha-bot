"""Thin wrapper around KiteConnect that:
1. Auto-resolves instrument_token from tradingsymbol+exchange
2. Pages historical_data calls to handle >100-day spans
"""
from __future__ import annotations
import logging, datetime as _dt
import pandas as pd
from kiteconnect import KiteConnect
from .config import load_config, KiteCreds

class KiteWrapper:
    """Utility functions on top of KiteConnect."""

    def __init__(self, creds: KiteCreds | None = None):
        if creds is None:
            creds = load_config()
        self.creds = creds

        # Initialise Kite client
        self.kite = KiteConnect(api_key=creds.api_key)
        self.kite.set_access_token(creds.access_token)

        # Auto-lookup instrument_token if missing
        if self.creds.instrument_token is None:
            # Fetch the complete instruments list, then filter
            insts = self.kite.instruments()

            # Narrow down to your exchange + equity type + symbol
            match = [
                i for i in insts
                if i.get("exchange") == creds.exchange
                and i.get("instrument_type") == "EQ"
                and i.get("tradingsymbol") == creds.tradingsymbol
            ]

            if not match:
                raise ValueError(
                    f"Symbol '{creds.tradingsymbol}' not found as EQ on {creds.exchange}"
                )

            self.creds.instrument_token = match[0]["instrument_token"]
            logging.info(
                "Auto-resolved %s on %s → instrument_token %s",
                creds.tradingsymbol,
                creds.exchange,
                self.creds.instrument_token,
            )

    def history(
        self,
        days: int,
        interval: str,
        instrument_token: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for up to `days` calendar days by slicing into
        100-day windows (Zerodha limit). Returns a DataFrame with
        columns: date, open, high, low, close, volume.
        """
        token = instrument_token or self.creds.instrument_token
        if token is None:
            raise ValueError("No instrument_token configured.")

        end_dt = _dt.datetime.now()
        start_dt = end_dt - _dt.timedelta(days=days)
        all_bars: list[dict] = []
        chunk_start = start_dt

        # Zerodha limits ~100 days per request
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + _dt.timedelta(days=100), end_dt)
            from_dt = chunk_start.replace(hour=9, minute=15, second=0, microsecond=0)
            to_dt   = chunk_end

            try:
                bars = self.kite.historical_data(
                    token,
                    from_dt,
                    to_dt,
                    interval,
                    continuous=False,
                    oi=False,
                )
            except Exception as e:
                logging.error("Failed to fetch %s→%s: %s", from_dt, to_dt, e)
                break

            all_bars.extend(bars)
            chunk_start = chunk_end

        df = pd.DataFrame.from_records(all_bars)
        df["date"] = pd.to_datetime(df["date"])
        return df
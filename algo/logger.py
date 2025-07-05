"""
logger.py – SQLite-backed trade logger with full fee & P/L reporting
(rev 2 – adds Order alias for backward compatibility)
"""

from __future__ import annotations
import sqlite3, json, datetime as _dt
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

DB_PATH = Path("trades.sqlite")

# ─── dataclasses ────────────────────────────────────────────────────
@dataclass(slots=True)
class OpenTrade:
    timestamp: _dt.datetime
    symbol: str
    side: str            # BUY / SELL
    qty: int
    price: float
    sl: float
    tp: float

@dataclass(slots=True)
class CloseTrade:
    timestamp: _dt.datetime
    symbol: str
    qty: int
    price: float
    reason: str          # TP / SL / TIME
    fee_detail: dict
    gross_pnl: float
    net_pnl: float

# ─── durable logger ─────────────────────────────────────────────────
class TradeLogger:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    # PUBLIC API -----------------------------------------------------
    def log_open(self, trade: OpenTrade):
        self.conn.execute(
            """INSERT INTO trades
               (timestamp,symbol,event,side,qty,price,sl,tp)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                trade.timestamp.isoformat(),
                trade.symbol,
                "OPEN",
                trade.side,
                trade.qty,
                trade.price,
                trade.sl,
                trade.tp,
            ),
        )
        self.conn.commit()

    def log_close(self, trade: CloseTrade):
        self.conn.execute(
            """INSERT INTO trades
               (timestamp,symbol,event,qty,price,reason,
                fee_json,gross_pnl,net_pnl)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                trade.timestamp.isoformat(),
                trade.symbol,
                "CLOSE",
                trade.qty,
                trade.price,
                trade.reason,
                json.dumps(trade.fee_detail, separators=(",", ":")),
                trade.gross_pnl,
                trade.net_pnl,
            ),
        )
        self.conn.commit()

    def summary(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM trades ORDER BY id", self.conn)

    # INTERNAL -------------------------------------------------------
    def _create_table(self):
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS trades (
               id         INTEGER PRIMARY KEY AUTOINCREMENT,
               timestamp  TEXT,
               symbol     TEXT,
               event      TEXT,
               side       TEXT,
               qty        INTEGER,
               price      REAL,
               sl         REAL,
               tp         REAL,
               reason     TEXT,
               fee_json   TEXT,
               gross_pnl  REAL,
               net_pnl    REAL
            )"""
        )
        self.conn.commit()

# ─── backward-compatibility alias ───────────────────────────────────
Order = OpenTrade      # old imports still work ⇢  from algo.logger import Order

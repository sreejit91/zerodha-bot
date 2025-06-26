"""Trade & event logger – SQLite‑backed.

Table **trades** schema
-----------------------
id          INTEGER  PK  autoincrement
timestamp   TEXT     ISO‑8601
symbol      TEXT     eg. "BANKNIFTY24JUNFUT" or "PnL"
side        TEXT     BUY / SELL / CLOSE
qty         INTEGER  number of lots / shares
price       REAL     fill price; 0 for PnL rows
pnl         REAL     closed‑trade PnL in rupees (optional)
broker_id   TEXT     Kite order‑id (nullable)
"""
from __future__ import annotations
import sqlite3, datetime as _dt
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

DB_PATH = Path("trades.sqlite")

@dataclass(slots=True)
class Order:
    timestamp: _dt.datetime
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: int
    price: float
    broker_id: str | None = None


class TradeLogger:
    """Minimal but robust trade logger."""

    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    # ---------------------------------------------------------------------
    #  Public helpers
    # ---------------------------------------------------------------------
    def log(self, order: Order):
        """Log a live trade (BUY / SELL)."""
        self.conn.execute(
            "INSERT INTO trades (timestamp, symbol, side, qty, price, broker_id) "
            "VALUES (?,?,?,?,?,?)",
            (
                order.timestamp.isoformat(),
                order.symbol,
                order.side,
                order.qty,
                order.price,
                order.broker_id,
            ),
        )
        self.conn.commit()

    def log_pnl(self, pnl: float):
        """Insert a synthetic row with the closed‑trade PnL (called on exit)."""
        self.conn.execute(
            "INSERT INTO trades (timestamp, symbol, side, qty, price, pnl, broker_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                _dt.datetime.now().isoformat(),
                "PnL",
                "CLOSE",
                0,
                0.0,
                pnl,
                None,
            ),
        )
        self.conn.commit()

    def summary(self) -> tuple[float, float]:
        """Return (total_PnL, win_rate%)."""
        df = pd.read_sql("SELECT * FROM trades WHERE pnl IS NOT NULL", self.conn)
        if df.empty:
            return 0.0, 0.0
        total = df.pnl.sum()
        win_rate = (df.pnl > 0).mean() * 100
        return float(total), float(win_rate)

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------
    def _create_table(self):
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS trades ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "timestamp TEXT,"
            "symbol TEXT,"
            "side TEXT,"
            "qty INTEGER,"
            "price REAL,"
            "pnl REAL,"
            "broker_id TEXT)"
        )
        self.conn.commit()
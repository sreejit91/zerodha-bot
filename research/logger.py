# algo/logger.py
import sqlite3, json
from dataclasses import dataclass

@dataclass
class OpenTrade:
    timestamp: str
    symbol: str
    side: str
    qty: int
    price: float
    sl: float
    tp: float

@dataclass
class CloseTrade:
    timestamp: str
    symbol: str
    qty: int
    price: float
    reason: str
    fee_detail: dict
    gross_pnl: float
    net_pnl: float

class TradeLogger:
    def __init__(self, db_path: str = "trades.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # create table if missing
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id    INTEGER,
                timestamp   TEXT NOT NULL,
                symbol      TEXT,
                side        TEXT,
                qty         INTEGER,
                price       REAL,
                sl          REAL,
                tp          REAL,
                event       TEXT,
                fee_detail  TEXT,
                gross_pnl   REAL,
                net_pnl     REAL
            )
        """)
        # ── schema-upgrade: add trade_id column if the DB predates it ──
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(trades)")}
        if "trade_id" not in cols:
            self.conn.execute("ALTER TABLE trades ADD COLUMN trade_id INTEGER")
        self.conn.commit()

    # ── logging helpers ─────────────────────────────────────────────
    def log_open(self, trade: OpenTrade) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades
               (timestamp,symbol,side,qty,price,sl,tp,event)
               VALUES (?,?,?,?,?,?,?,'open')""",
            (
                trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else trade.timestamp,
                trade.symbol, trade.side, trade.qty, trade.price, trade.sl, trade.tp
            )
        )
        self.conn.commit()
        return cur.lastrowid           # returned trade_id

    def log_close(self, trade: CloseTrade, trade_id: int | None):
        if trade_id is None:
            return                      # safety: no matching open
        self.conn.execute(
            """INSERT INTO trades
               (trade_id,timestamp,symbol,qty,price,event,
                fee_detail,gross_pnl,net_pnl)
               VALUES (?,?,?,?,?,'close',?,?,?)""",
            (
                trade_id,
                trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else trade.timestamp,
                trade.symbol, trade.qty, trade.price,
                json.dumps(trade.fee_detail),
                trade.gross_pnl, trade.net_pnl
            )
        )
        self.conn.commit()

# backward-compat alias
Order = OpenTrade

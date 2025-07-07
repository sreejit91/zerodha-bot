import sqlite3
import json
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
        # open a connection, allow logging from multiple threads
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # create table with an 'event' column to distinguish open/close
        self.conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                qty INTEGER,
                price REAL,
                sl REAL,
                tp REAL,
                event TEXT,
                fee_detail TEXT,
                gross_pnl REAL,
                net_pnl REAL
            )
            '''
        )
        self.conn.commit()

    def log_open(self, trade: OpenTrade):
        # insert an open event
        self.conn.execute(
            '''
            INSERT OR IGNORE INTO trades
            (timestamp, symbol, side, qty, price, sl, tp, event)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else trade.timestamp,
             trade.symbol, trade.side, trade.qty, trade.price, trade.sl, trade.tp, 'open')
        )
        self.conn.commit()

    def log_close(self, trade: CloseTrade):
        # update existing open row or insert if missing, marking close event and adding PnL
        self.conn.execute(
            '''
            INSERT OR REPLACE INTO trades
            (timestamp, symbol, qty, price, event, fee_detail, gross_pnl, net_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else trade.timestamp,
             trade.symbol, trade.qty, trade.price, 'close', json.dumps(trade.fee_detail),
             trade.gross_pnl, trade.net_pnl)
        )
        self.conn.commit()
# backward-compat alias for Order import in broker/__init__ and elsewhere
Order = OpenTrade

# algo/config.py

import json
from dataclasses import dataclass
from pathlib import Path

# Look for config.json in the same directory as this file
CONFIG_PATH = Path(__file__).parent / "config.json"

@dataclass
class KiteCreds:
    api_key: str
    api_secret: str | None
    access_token: str
    tradingsymbol: str
    exchange: str
    instrument_token: int | None = None

def load_config() -> KiteCreds:
    data = json.loads(CONFIG_PATH.read_text())
    return KiteCreds(
        api_key        = data["api_key"],
        api_secret     = data.get("api_secret"),
        access_token   = data["access_token"],
        tradingsymbol  = data["tradingsymbol"],
        exchange       = data.get("exchange", "NSE"),
        instrument_token=None,
    )

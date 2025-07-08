from pathlib import Path
import json
from dataclasses import dataclass

MAIN_CFG = Path(__file__).resolve().parent.parent / "config.json"

@dataclass
class KiteCreds:
    api_key: str
    api_secret: str | None
    access_token: str
    tradingsymbol: str
    exchange: str

_EXPECTED = set(KiteCreds.__annotations__)      # {'api_key', 'api_secret', …}

def load_config() -> KiteCreds:
    data = json.loads(MAIN_CFG.read_text())

    # keep only the expected keys
    filtered = {k: v for k, v in data.items() if k in _EXPECTED}

    # research-only overrides
    filtered["tradingsymbol"] = "RELIANCE"
    filtered["exchange"]      = "NSE"

    return KiteCreds(**filtered)

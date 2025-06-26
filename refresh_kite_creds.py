# refresh_kite_creds.py
"""
Run this script in a standard terminal (not inside Jupyter) each morning before running your notebook.
It:
  1. Prompts you to open the login URL and paste the request_token
  2. Generates a fresh access_token
  3. Rolls the BANKNIFTY futures instrument_token to the front-month
  4. Updates config.json with both tokens
"""
import json
import pathlib
import pandas as pd
from kiteconnect import KiteConnect

CONFIG_PATH = pathlib.Path(__file__).parent / "config.json"

def load_creds():
    return json.loads(CONFIG_PATH.read_text())

def save_creds(creds):
    CONFIG_PATH.write_text(json.dumps(creds, indent=2))
    print("config.json updated successfully.")

def refresh_access_token(creds):
    kite = KiteConnect(api_key=creds["api_key"])
    print("1. Open this URL in your browser, log in, then copy the `request_token` from the URL:")
    print(kite.login_url())
    req_token = input("Paste the request_token here: ")
    data = kite.generate_session(req_token.strip(), creds["api_secret"])
    creds["access_token"] = data["access_token"]
    print("Access token refreshed.")

def roll_banknifty_token(creds):
    kite = KiteConnect(api_key=creds["api_key"])
    kite.set_access_token(creds["access_token"])
    instruments = pd.DataFrame(kite.instruments("NFO"))
    futs = instruments[
        (instruments.segment == "NFO-FUT") & instruments.tradingsymbol.str.startswith("BANKNIFTY")
    ].copy()
    futs["expiry"] = pd.to_datetime(futs["expiry"])
    today = pd.Timestamp.today().normalize()
    front = futs[futs["expiry"] >= today].sort_values("expiry").iloc[0]
    creds["instrument_token"] = int(front["instrument_token"])
    print(f"Rolled to {front['tradingsymbol']} (token {creds['instrument_token']})")

if __name__ == "__main__":
    creds = load_creds()
    refresh_access_token(creds)
    roll_banknifty_token(creds)
    save_creds(creds)

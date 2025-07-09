# refresh_kite_creds.py

"""
Run this script each morning to refresh your Kite access token.
It:
  1. Prompts you to open the login URL and paste the request_token
  2. Generates a fresh access_token
  3. Updates config.json with the new token
"""

import json
import pathlib
from kiteconnect import KiteConnect

CONFIG_PATH = pathlib.Path(__file__).parent / "config.json"

def load_creds():
    """Read and return the credentials dict from config.json."""
    return json.loads(CONFIG_PATH.read_text())

def save_creds(creds):
    """Write the updated credentials back to config.json."""
    CONFIG_PATH.write_text(json.dumps(creds, indent=2))
    print("config.json updated successfully.")

def refresh_access_token(creds):
    """Generate a new access token via KiteConnect and update the creds dict."""
    kite = KiteConnect(api_key=creds["api_key"])
    print("1. Open this URL in your browser, log in, then paste the `request_token` from the URL:")
    print(kite.login_url())
    req_token = input("Paste the request_token here: ").strip()
    data = kite.generate_session(req_token, creds["api_secret"])
    creds["access_token"] = data["access_token"]
    print("Access token refreshed.")

if __name__ == "__main__":
    creds = load_creds()
    refresh_access_token(creds)
    save_creds(creds)

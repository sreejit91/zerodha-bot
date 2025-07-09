#!/usr/bin/env python3
from algo.refresh_kite_creds import load_creds, save_creds, refresh_access_token

def main():
    """
    Simple one-click script to refresh your Kite access token once per day.
    """
    creds = load_creds()
    refresh_access_token(creds)
    save_creds(creds)
    print("✅ Access token refreshed and config.json updated.")

if __name__ == "__main__":
    main()

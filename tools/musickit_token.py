"""Sign the Apple Music developer token the demo page plays Apple Music tracks with.

Usage: uv run --with pyjwt --with cryptography python tools/musickit_token.py \
           --key AuthKey_XXXXXXXXXX.p8 --key-id XXXXXXXXXX --team-id XXXXXXXXXX

The key is a MusicKit private key from the Apple Developer account
(Certificates, Identifiers and Profiles, Keys, with MusicKit enabled). It stays
on this Mac and is never committed. The token it prints is public by design:
MusicKit on the web needs it in the page. It is limited to the demo page's
origin and lasts the six months Apple allows, so sign a new one before it ends.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

ORIGIN = "https://jtannahill.github.io"
SIX_MONTHS = 15_777_000  # seconds, Apple's limit


def main() -> None:
    import jwt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--origin", action="append", help=f"allowed origin (default {ORIGIN}); repeat for more")
    args = parser.parse_args()
    now = int(time.time())
    claims = {"iss": args.team_id, "iat": now, "exp": now + SIX_MONTHS, "origin": args.origin or [ORIGIN]}
    print(jwt.encode(claims, args.key.read_text(), algorithm="ES256", headers={"kid": args.key_id}))


if __name__ == "__main__":
    main()

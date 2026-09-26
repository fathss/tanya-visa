"""Politely fetch source documents into data/raw_docs/.

Policy:
  - robots.txt is fetched per host and consulted before every URL;
    disallowed paths are skipped, never fetched.
  - A delay of 1.5 s is kept between requests.
  - Raw HTML is vendored as-is; loader.py extracts the content root.

Usage:
    python scripts/fetch_docs.py scripts/fetch_manifest.txt [--dry-run] [--delay 1.5]

Manifest format, one target per line:
    <output_filename><TAB><url>
Blank lines and lines starting with '#' are ignored. Any run of whitespace
between the two fields is accepted, so space-aligned lines are fine too.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

USER_AGENT = "hacktiv-rag-corpus-bot/1.0 (educational RAG corpus; +https://github.com/)"
TIMEOUT = 30

_last_request_at = 0.0


def _throttle(delay: float) -> None:
    global _last_request_at
    wait = delay - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _robots(host_url: str, session: requests.Session, delay: float) -> urllib.robotparser.RobotFileParser:
    parser = urllib.robotparser.RobotFileParser()
    robots_url = host_url.rstrip("/") + "/robots.txt"
    _throttle(delay)
    try:
        response = session.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    except requests.RequestException as error:
        print(f"  [robots] {robots_url} unreachable ({error}); skipping host conservatively")
        parser.parse(["User-agent: *", "Disallow: /"])
        return parser

    if response.status_code == 404:
        print(f"  [robots] {robots_url} -> 404 (not a prohibition); allow-all")
        parser.parse([])
    elif response.status_code >= 400:
        print(f"  [robots] {robots_url} -> HTTP {response.status_code}; skipping host conservatively")
        parser.parse(["User-agent: *", "Disallow: /"])
    else:
        parser.parse(response.text.splitlines())
        print(f"  [robots] {robots_url} -> HTTP 200")
    return parser


def _read_manifest(path: Path) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{number}: expected '<filename> <url>' separated by a tab or spaces")
        targets.append((parts[0], parts[1]))
    return targets


def _extension(url: str, content_type: str) -> str:
    if url.lower().endswith(".pdf") or "application/pdf" in content_type:
        return ".pdf"
    return ".html"


def fetch(manifest_path: Path, dry_run: bool, delay: float) -> int:
    targets = _read_manifest(manifest_path)
    session = requests.Session()
    robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
    failures = 0

    for filename, url in targets:
        parsed = urlparse(url)
        host_url = f"{parsed.scheme}://{parsed.netloc}"
        print(f"\n{filename}\n  <- {url}")

        if host_url not in robots_cache:
            robots_cache[host_url] = _robots(host_url, session, delay)
        parser = robots_cache[host_url]

        if not parser.can_fetch(USER_AGENT, url):
            print(f"  [skip] disallowed by robots.txt")
            failures += 1
            continue

        if dry_run:
            print("  [dry-run] allowed; not fetching")
            continue

        _throttle(delay)
        try:
            response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        except requests.RequestException as error:
            print(f"  [fail] {error}")
            failures += 1
            continue

        if response.status_code >= 400:
            print(f"  [fail] HTTP {response.status_code}")
            failures += 1
            continue

        content_type = response.headers.get("Content-Type", "")
        extension = _extension(url, content_type)
        target = (config.RAW_DOCS_DIR / filename).with_suffix(extension)

        if extension == ".pdf":
            target.write_bytes(response.content)
        else:
            target.write_text(response.text, encoding="utf-8")
        print(f"  [ok] HTTP {response.status_code} {content_type.split(';')[0]} -> {target.name} ({len(response.content):,} bytes)")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="file with '<filename><TAB><url>' lines")
    parser.add_argument("--dry-run", action="store_true", help="only check robots.txt, fetch nothing")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between requests (default 1.5)")
    args = parser.parse_args()

    failures = fetch(args.manifest, args.dry_run, args.delay)
    print(f"\nDone. {failures} target(s) skipped or failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

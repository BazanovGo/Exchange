"""Download continuous futures candles from MOEX AlgoPack (apim.moex.com).

Builds the series stored in ``data/raw/moex_algopack``: contracts are glued
by nearest expiry (a contract is held through its last trading day, the next
one starts the following day, or right after the last candle if the contract
stopped trading earlier), prices are not adjusted. 1m and 60m candles come
from the API, 5m are aggregated from 1m.

    MOEX_ALGOPACK_TOKEN=... python scripts/download_moex_algopack.py \
        --assets GOLD SILV --ca-bundle russian_trusted_root_ca.pem

apim.moex.com is signed by the Russian Trusted Root CA, which is missing from
most trust stores: pass the official certificate via ``--ca-bundle``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

API = "https://apim.moex.com/iss"
PAGE = 500
MONTH_CODES = "FGHJKMNQUVXZ"
COLUMNS = ["ticker", "secid", "begin", "end", "open", "high", "low", "close", "volume", "value"]
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "moex_algopack"


class Client:
    def __init__(self, token: str, ca_bundle: str | bool = True):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.verify = ca_bundle

    def get(self, path: str, **params) -> dict:
        params["iss.meta"] = "off"
        for attempt in range(8):
            try:
                r = self.session.get(f"{API}{path}", params=params, timeout=120, verify=self.verify)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (401, 403):
                    raise RuntimeError(f"{r.status_code} {path}: {r.text[:200]}")
            except (requests.ConnectionError, requests.Timeout, ValueError):
                pass
            time.sleep(min(2 ** attempt, 60))
        raise RuntimeError(f"giving up on {path} {params}")


def describe(client: Client, secid: str) -> dict:
    data = client.get(f"/securities/{secid}.json", **{"iss.only": "description"})["description"]
    return {r[0]: r[2] for r in data["data"]}


def contracts(client: Client, asset: str, workers: int) -> list[dict]:
    """Outright contracts of an asset (calendar spreads dropped), by expiry.

    The series list misses part of the expired contracts, so every month absent
    from it is probed by secid: ``BRQ3`` or, once the code was reused,
    ``BRQ3_2013``; a candidate counts only if its short name matches.
    """
    data = client.get("/statistics/engines/futures/markets/forts/series.json",
                      asset_code=asset, show_expired=1)["series"]
    rows = [dict(zip(data["columns"], r)) for r in data["data"]]
    pattern = re.compile(rf"{re.escape(asset)}-\d+\.\d+", re.IGNORECASE)
    rows = [r for r in rows if pattern.fullmatch(r["name"])]

    known = {r["name"].upper() for r in rows}
    prefix = Counter(r["secid"][:2] for r in rows).most_common(1)[0][0]
    first = int(min(r["start_date"] for r in rows)[:4])
    last = int(max(r["expiration_date"] for r in rows)[:4])
    candidates = []
    for year in range(first, last + 1):
        for month in range(1, 13):
            name = f"{asset}-{month}.{year % 100:02d}"
            if name.upper() not in known:
                code = f"{prefix}{MONTH_CODES[month - 1]}{year % 10}"
                candidates += [(name, code), (name, f"{code}_{year}")]
    with ThreadPoolExecutor(workers) as pool:
        found = pool.map(lambda c: (c[0], describe(client, c[1])), candidates)
        for name, desc in found:
            if desc.get("SHORTNAME", "").upper() == name.upper() and name.upper() not in known:
                known.add(name.upper())
                rows.append({"secid": desc["SECID"], "name": name,
                             "start_date": desc["FRSTTRADE"], "expiration_date": desc["LSTTRADE"]})
    return sorted(rows, key=lambda r: r["expiration_date"])


def windows(rows: list[dict], start: date, till: date) -> list[tuple[str, date, date]]:
    """(secid, first_day, last_day) of each contract in the continuous series."""
    out, prev = [], None
    for r in rows:
        exp = date.fromisoformat(r["expiration_date"])
        first = prev + timedelta(days=1) if prev else date.fromisoformat(r["start_date"])
        prev = exp
        lo, hi = max(first, start), min(exp, till)
        if lo <= hi:
            out.append((r["secid"], lo, hi))
    return out


def month_chunks(lo: date, hi: date):
    while lo <= hi:
        nxt = (lo.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield lo, min(hi, nxt - timedelta(days=1))
        lo = nxt


def fetch(client: Client, secid: str, interval: int, lo: date, hi: date) -> list[list]:
    path = f"/engines/futures/markets/forts/securities/{secid}/candles.json"
    rows, start = [], 0
    while True:
        block = client.get(path, interval=interval, start=start,
                           **{"from": lo.isoformat(), "till": hi.isoformat()})["candles"]
        cols = block["columns"]
        page = [dict(zip(cols, r)) for r in block["data"]]
        rows += page
        if len(page) < PAGE:
            break
        start += len(page)
    lo_s, hi_s = lo.isoformat(), hi.isoformat()
    return [[secid, r["begin"], r["end"], r["open"], r["high"], r["low"], r["close"],
             r["volume"], r["value"]]
            for r in rows if lo_s <= r["begin"][:10] <= hi_s]


def fetch_all(client: Client, jobs: list[tuple], workers: int, label: str = "") -> dict[tuple, list]:
    results = {}
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, client, *job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if label and (i % 50 == 0 or i == len(jobs)):
                print(f"  {label}: {i}/{len(jobs)} chunks", flush=True)
    return results


def aggregate_5m(rows: list[list]) -> list[list]:
    out: list[list] = []
    for secid, begin, _, o, h, l, c, v, val in rows:
        t = datetime.fromisoformat(begin)
        b = t.replace(minute=t.minute - t.minute % 5, second=0)
        key = b.strftime("%Y-%m-%d %H:%M:%S")
        if out and out[-1][1] == key and out[-1][0] == secid:
            bar = out[-1]
            bar[4], bar[5], bar[6] = max(bar[4], h), min(bar[5], l), c
            bar[7] += v
            bar[8] += val
        else:
            end = (b + timedelta(minutes=4, seconds=59)).strftime("%Y-%m-%d %H:%M:%S")
            out.append([secid, key, end, o, h, l, c, v, val])
    return out


def write_years(ticker: str, tf: str, rows: list[list], out_dir: Path) -> dict[str, int]:
    by_year: dict[str, list] = defaultdict(list)
    for r in rows:
        by_year[r[1][:4]].append(r)
    written = {}
    for year, yrows in sorted(by_year.items()):
        path = out_dir / ticker / tf / f"{year}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows([ticker, *r] for r in yrows)
        written[f"{ticker}/{tf}/{year}.csv"] = len(yrows)
    return written


def download_asset(client: Client, asset: str, start: date, till: date,
                   out_dir: Path, workers: int) -> dict[str, int]:
    ticker = asset.lower()
    spans = windows(contracts(client, asset, workers), start, till)
    jobs = [(secid, interval, lo, hi)
            for secid, a, b in spans for lo, hi in month_chunks(a, b) for interval in (1, 60)]
    results = fetch_all(client, jobs, workers, asset)
    candles: dict[tuple, list] = defaultdict(list)
    for job in jobs:
        candles[job[:2]] += results[job]

    # A contract may stop trading before its last trading day: the next one
    # then takes over the day after the last candle, not after the expiry.
    last = None
    for secid, lo, _ in spans:
        if last and last + timedelta(days=1) < lo:
            gap = [(secid, interval, a, b) for a, b in month_chunks(last + timedelta(days=1),
                                                                    lo - timedelta(days=1))
                   for interval in (1, 60)]
            extra = fetch_all(client, gap, workers)
            for interval in (1, 60):
                candles[secid, interval] = [r for job in gap if job[1] == interval
                                            for r in extra[job]] + candles[secid, interval]
        if candles[secid, 1]:
            last = date.fromisoformat(candles[secid, 1][-1][1][:10])

    series = {interval: [r for secid, _, _ in spans for r in candles[secid, interval]]
              for interval in (1, 60)}
    for rows in series.values():
        rows.sort(key=lambda r: r[1])
    written = {}
    written.update(write_years(ticker, "1m", series[1], out_dir))
    written.update(write_years(ticker, "5m", aggregate_5m(series[1]), out_dir))
    written.update(write_years(ticker, "60m", series[60], out_dir))
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets", nargs="+", required=True, help="MOEX asset codes: NG BR WTI GOLD ...")
    p.add_argument("--from", dest="start", default="1900-01-01")
    p.add_argument("--till", default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--ca-bundle", default=os.environ.get("MOEX_CA_BUNDLE"))
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    token = os.environ.get("MOEX_ALGOPACK_TOKEN")
    if not token:
        print("MOEX_ALGOPACK_TOKEN is not set", file=sys.stderr)
        return 1
    client = Client(token, args.ca_bundle or True)
    start, till = date.fromisoformat(args.start), date.fromisoformat(args.till)

    manifest_path = args.out_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists() else
                {"mode": "continuous-nearest-expiry", "from": args.start,
                 "timezone": "Europe/Moscow", "files": {}})
    for asset in args.assets:
        print(f"{asset}: downloading {start}..{till}", flush=True)
        written = download_asset(client, asset, start, till, args.out_dir, args.workers)
        manifest["files"].update(written)
        print(f"{asset}: {sum(v for k, v in written.items() if '/1m/' in k)} 1m candles", flush=True)
    manifest["till"] = args.till
    manifest["files"] = dict(sorted(manifest["files"].items()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

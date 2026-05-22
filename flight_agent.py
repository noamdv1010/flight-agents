"""Autonomous flight-deal hunter.

Pipeline (per run):
    config.yaml + secrets
        -> Skyscanner (sky-scrapper @ RapidAPI) per destination & date sample
        -> normalize -> hard price filter -> dedupe against state.json
        -> Claude Haiku 4.5 scores each survivor (1-10) via tool-use JSON
        -> Telegram alert for any score >= min_score
        -> persist state.json (committed back by GitHub Actions)

CLI:
    python flight_agent.py                # production run
    python flight_agent.py --dry-run      # no Telegram, prints decisions
    python flight_agent.py --force-alert  # send a synthetic Telegram ping (E2E test)
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
import yaml
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
STATE_PATH = SCRIPT_DIR / "state.json"
AIRPORT_CACHE_PATH = SCRIPT_DIR / "airport_cache.json"

RAPIDAPI_HOST = "sky-scrapper.p.rapidapi.com"
SEARCH_FLIGHTS_URL = f"https://{RAPIDAPI_HOST}/api/v2/flights/searchFlights"
SEARCH_AIRPORT_URL = f"https://{RAPIDAPI_HOST}/api/v1/flights/searchAirport"

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
STATE_TTL_DAYS = 14

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("flight-agent")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FlightDeal:
    origin: str
    destination: str
    depart_date: str            # YYYY-MM-DD
    return_date: str            # YYYY-MM-DD
    price_usd: float
    total_duration_minutes: int  # outbound + inbound combined
    stops_outbound: int
    stops_inbound: int
    layover_minutes: int         # max single layover across both directions
    carriers: list[str]
    deep_link: str
    cabin_class: str = "economy"

    def fingerprint(self) -> str:
        """Stable hash for dedupe. Price bucketed to nearest $10."""
        bucket = round(self.price_usd / 10) * 10
        raw = f"{self.origin}|{self.destination}|{self.depart_date}|{self.return_date}|{bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def trip_length(self) -> int:
        d1 = datetime.strptime(self.depart_date, "%Y-%m-%d")
        d2 = datetime.strptime(self.return_date, "%Y-%m-%d")
        return (d2 - d1).days


@dataclass
class Config:
    origin: str
    currency: str
    min_score: int
    trip_length_days: tuple[int, int]
    date_window_days: int
    samples_per_destination: int
    cabin_class: str
    adults: int
    destinations: list[dict[str, Any]] = field(default_factory=list)

    # Secrets
    rapidapi_key: str = ""
    anthropic_key: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""


# ---------------------------------------------------------------------------
# Config & state
# ---------------------------------------------------------------------------


def load_config() -> Config:
    if not CONFIG_PATH.exists():
        sys.exit(f"missing config: {CONFIG_PATH}")
    raw = yaml.safe_load(CONFIG_PATH.read_text())

    def need_env(name: str) -> str:
        val = os.environ.get(name, "").strip()
        if not val:
            sys.exit(f"missing required env var: {name}")
        return val

    return Config(
        origin=raw["origin"],
        currency=raw.get("currency", "USD"),
        min_score=int(raw.get("min_score", 8)),
        trip_length_days=tuple(raw.get("trip_length_days", [4, 14])),
        date_window_days=int(raw.get("date_window_days", 120)),
        samples_per_destination=int(raw.get("samples_per_destination", 6)),
        cabin_class=raw.get("cabin_class", "economy"),
        adults=int(raw.get("adults", 1)),
        destinations=raw["destinations"],
        rapidapi_key=need_env("RAPIDAPI_KEY"),
        anthropic_key=need_env("ANTHROPIC_API_KEY"),
        telegram_token=need_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=need_env("TELEGRAM_CHAT_ID"),
    )


def load_state() -> dict[str, str]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("state.json corrupt, starting fresh")
        return {}


def save_state(state: dict[str, str]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def prune_state(state: dict[str, str]) -> dict[str, str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=STATE_TTL_DAYS)
    return {
        k: v for k, v in state.items()
        if datetime.fromisoformat(v) > cutoff
    }


# ---------------------------------------------------------------------------
# Sky-scrapper integration
# ---------------------------------------------------------------------------


def _rapidapi_headers(key: str) -> dict[str, str]:
    return {"x-rapidapi-key": key, "x-rapidapi-host": RAPIDAPI_HOST}


def _load_airport_cache() -> dict[str, dict[str, str]]:
    if AIRPORT_CACHE_PATH.exists():
        try:
            return json.loads(AIRPORT_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save_airport_cache(cache: dict[str, dict[str, str]]) -> None:
    AIRPORT_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def resolve_airport(iata: str, key: str, cache: dict[str, dict[str, str]]) -> dict[str, str] | None:
    """Sky-scrapper requires its own skyId/entityId pair, not raw IATA codes."""
    iata = iata.upper()
    if iata in cache:
        return cache[iata]

    try:
        r = requests.get(
            SEARCH_AIRPORT_URL,
            headers=_rapidapi_headers(key),
            params={"query": iata, "locale": "en-US"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        for item in data:
            nav = item.get("navigation", {})
            entity_type = nav.get("entityType", "").upper()
            if entity_type != "AIRPORT":
                continue
            flight_params = nav.get("relevantFlightParams", {})
            sky_id = flight_params.get("skyId") or item.get("skyId")
            entity_id = flight_params.get("entityId") or item.get("entityId")
            if sky_id and sky_id.upper() == iata:
                cache[iata] = {"skyId": sky_id, "entityId": str(entity_id)}
                _save_airport_cache(cache)
                return cache[iata]
        # fallback to first AIRPORT result if exact skyId didn't match
        for item in data:
            nav = item.get("navigation", {})
            if nav.get("entityType", "").upper() != "AIRPORT":
                continue
            fp = nav.get("relevantFlightParams", {})
            if fp.get("skyId") and fp.get("entityId"):
                cache[iata] = {"skyId": fp["skyId"], "entityId": str(fp["entityId"])}
                _save_airport_cache(cache)
                return cache[iata]
    except requests.RequestException as e:
        log.warning("resolve_airport %s failed: %s", iata, e)
    return None


def search_flights(
    cfg: Config,
    origin_air: dict[str, str],
    dest_air: dict[str, str],
    depart: str,
    return_: str,
) -> list[dict[str, Any]] | None:
    params = {
        "originSkyId": origin_air["skyId"],
        "destinationSkyId": dest_air["skyId"],
        "originEntityId": origin_air["entityId"],
        "destinationEntityId": dest_air["entityId"],
        "date": depart,
        "returnDate": return_,
        "cabinClass": cfg.cabin_class,
        "adults": str(cfg.adults),
        "sortBy": "best",
        "currency": cfg.currency,
        "market": "en-US",
        "countryCode": "US",
    }

    for attempt in (1, 2):
        try:
            r = requests.get(
                SEARCH_FLIGHTS_URL,
                headers=_rapidapi_headers(cfg.rapidapi_key),
                params=params,
                timeout=30,
            )
            if r.status_code in (429, 500, 502, 503, 504) and attempt == 1:
                log.warning("rapidapi %s, retrying once", r.status_code)
                time.sleep(3)
                continue
            r.raise_for_status()
            payload = r.json()
            if not payload.get("status"):
                log.warning("sky-scrapper soft-fail: %s", payload.get("message"))
                return None
            return payload.get("data", {}).get("itineraries", []) or []
        except requests.RequestException as e:
            if attempt == 2:
                log.warning("search_flights %s->%s failed: %s",
                            origin_air["skyId"], dest_air["skyId"], e)
                return None
            time.sleep(3)
    return None


def normalize(
    itineraries: list[dict[str, Any]],
    origin: str,
    destination: str,
    depart: str,
    return_: str,
    cabin: str,
) -> list[FlightDeal]:
    deals: list[FlightDeal] = []
    for it in itineraries:
        try:
            price = float(it.get("price", {}).get("raw", 0))
            if price <= 0:
                continue
            legs = it.get("legs", [])
            if len(legs) < 2:
                continue  # expecting round-trip
            out_leg, in_leg = legs[0], legs[1]
            total_dur = int(out_leg.get("durationInMinutes", 0)) + int(in_leg.get("durationInMinutes", 0))

            def max_layover(leg: dict[str, Any]) -> int:
                segments = leg.get("segments", [])
                if len(segments) < 2:
                    return 0
                gaps = []
                for a, b in zip(segments, segments[1:]):
                    try:
                        t1 = datetime.fromisoformat(a["arrival"])
                        t2 = datetime.fromisoformat(b["departure"])
                        gaps.append(int((t2 - t1).total_seconds() // 60))
                    except (KeyError, ValueError):
                        continue
                return max(gaps) if gaps else 0

            layover = max(max_layover(out_leg), max_layover(in_leg))

            carriers = set()
            for leg in legs:
                for c in leg.get("carriers", {}).get("marketing", []):
                    if c.get("name"):
                        carriers.add(c["name"])

            deep_link = (
                f"https://www.skyscanner.net/transport/flights/"
                f"{origin.lower()}/{destination.lower()}/"
                f"{depart.replace('-', '')[2:]}/{return_.replace('-', '')[2:]}/"
            )

            deals.append(FlightDeal(
                origin=origin,
                destination=destination,
                depart_date=depart,
                return_date=return_,
                price_usd=price,
                total_duration_minutes=total_dur,
                stops_outbound=int(out_leg.get("stopCount", 0)),
                stops_inbound=int(in_leg.get("stopCount", 0)),
                layover_minutes=layover,
                carriers=sorted(carriers),
                deep_link=deep_link,
                cabin_class=cabin,
            ))
        except (KeyError, TypeError, ValueError) as e:
            log.debug("skipping malformed itinerary: %s", e)
            continue
    return deals


# ---------------------------------------------------------------------------
# Date sampling
# ---------------------------------------------------------------------------


def sample_date_pairs(cfg: Config, seed_salt: str) -> list[tuple[str, str]]:
    """Pick N (depart, return) pairs across the rolling window.

    Uses a seed derived from today + destination so each destination gets a
    different sample but the same destination on the same day is stable.
    """
    rng = random.Random(f"{datetime.now(timezone.utc).date()}|{seed_salt}")
    today = datetime.now(timezone.utc).date()
    earliest = today + timedelta(days=14)            # skip too-close fares
    latest = today + timedelta(days=cfg.date_window_days)
    span = (latest - earliest).days
    if span <= 0:
        return []

    min_trip, max_trip = cfg.trip_length_days
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(pairs) < cfg.samples_per_destination and attempts < 50:
        attempts += 1
        offset = rng.randint(0, span)
        trip = rng.randint(min_trip, max_trip)
        d1 = earliest + timedelta(days=offset)
        d2 = d1 + timedelta(days=trip)
        if d2 > latest:
            continue
        key = (d1.isoformat(), d2.isoformat())
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


# ---------------------------------------------------------------------------
# LLM scoring (Claude Haiku 4.5 + tool use + prompt caching)
# ---------------------------------------------------------------------------


SCORING_SYSTEM = """You are a meticulous flight deal evaluator. You rate round-trip
itineraries on a 1-10 scale where:

  10 = mistake fare or extraordinary deal; book within the hour
   9 = excellent deal, well below typical, comfortable schedule
   8 = clearly good deal worth alerting the user about
   7 = decent price but with friction (very long layover, awkward times, weak carrier)
   1-6 = not worth interrupting the user

You weigh these factors:
  * Price vs. a reasonable baseline for the route and season
  * Total trip duration (each-way travel time)
  * Number of stops; non-stop is a strong positive
  * Longest single layover (anything > 5h is friction, > 10h is bad)
  * Carrier reputation — major full-service carriers and reputable LCCs are fine;
    obscure no-name carriers should drag the score down
  * Trip length matching what a real traveler would want

Be strict: most fares should NOT score 8+. Reserve high scores for genuine deals.
Return ONLY through the `record_deal_score` tool. Keep reasoning under 200 chars."""


SCORING_TOOL = {
    "name": "record_deal_score",
    "description": "Record the evaluation of one flight deal.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Overall deal quality on a 1-10 scale.",
            },
            "reasoning": {
                "type": "string",
                "maxLength": 220,
                "description": "One-sentence rationale that names the dominant factors.",
            },
        },
        "required": ["score", "reasoning"],
    },
}


def score_deal(
    client: Anthropic,
    deal: FlightDeal,
    max_price_for_route: float,
) -> tuple[int, str]:
    user_msg = (
        f"Route: {deal.origin} <-> {deal.destination}\n"
        f"Dates: {deal.depart_date} to {deal.return_date} ({deal.trip_length()} nights)\n"
        f"Price: ${deal.price_usd:.0f} USD (user's alert ceiling for this route: ${max_price_for_route:.0f})\n"
        f"Cabin: {deal.cabin_class}\n"
        f"Total travel time (both directions combined): "
        f"{deal.total_duration_minutes // 60}h {deal.total_duration_minutes % 60}m\n"
        f"Stops: outbound={deal.stops_outbound}, inbound={deal.stops_inbound}\n"
        f"Longest single layover: {deal.layover_minutes // 60}h {deal.layover_minutes % 60}m\n"
        f"Carriers: {', '.join(deal.carriers) or 'unknown'}\n\n"
        f"Score this deal."
    )

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        system=[{
            "type": "text",
            "text": SCORING_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[SCORING_TOOL],
        tool_choice={"type": "tool", "name": "record_deal_score"},
        messages=[{"role": "user", "content": user_msg}],
    )

    for block in resp.content:
        if block.type == "tool_use" and block.name == "record_deal_score":
            inp = block.input
            return int(inp["score"]), str(inp.get("reasoning", ""))
    return 0, "no tool output"


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def send_telegram(cfg: Config, text: str) -> None:
    """Send a Telegram message in HTML parse mode.

    HTML mode is much more forgiving than legacy Markdown — only `<`, `>`, `&`
    are special, and Telegram won't choke on identifiers containing `_` or `*`.
    Falls back to plain text if HTML parsing somehow still fails.
    """
    url = f"https://api.telegram.org/bot{cfg.telegram_token}/sendMessage"
    payload = {
        "chat_id": cfg.telegram_chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 400:
            log.warning("telegram HTML parse rejected, retrying as plain text: %s", r.text)
            payload.pop("parse_mode", None)
            # Strip HTML tags for the fallback
            import re
            payload["text"] = re.sub(r"<[^>]+>", "", text)
            r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        log.error("telegram send failed: %s", e)


def format_alert(deal: FlightDeal, score: int, reasoning: str) -> str:
    """Build an HTML-formatted Telegram message. All dynamic strings are escaped."""
    hrs = deal.total_duration_minutes // 60
    mins = deal.total_duration_minutes % 60
    lay_h = deal.layover_minutes // 60
    lay_m = deal.layover_minutes % 60
    stops = f"{deal.stops_outbound}+{deal.stops_inbound} stops"
    carriers = ", ".join(deal.carriers[:3]) or "unknown"

    e = html.escape  # all user/api-controlled strings flow through this
    return (
        f"<b>Gold Deal {score}/10</b> — {e(deal.origin)} → {e(deal.destination)}\n\n"
        f"<b>Price:</b> ${deal.price_usd:.0f}\n"
        f"<b>Dates:</b> {e(deal.depart_date)} → {e(deal.return_date)} "
        f"({deal.trip_length()}n)\n"
        f"<b>Travel time:</b> {hrs}h{mins:02d}m total, {stops}, "
        f"longest layover {lay_h}h{lay_m:02d}m\n"
        f"<b>Carriers:</b> {e(carriers)}\n\n"
        f"<i>{e(reasoning)}</i>\n\n"
        f'<a href="{e(deal.deep_link, quote=True)}">Open on Skyscanner</a>'
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(cfg: Config, dry_run: bool) -> None:
    client = Anthropic(api_key=cfg.anthropic_key)
    state = prune_state(load_state())
    airport_cache = _load_airport_cache()

    origin_air = resolve_airport(cfg.origin, cfg.rapidapi_key, airport_cache)
    if not origin_air:
        sys.exit(f"could not resolve origin airport {cfg.origin}")

    total_seen = 0
    total_alerted = 0

    for dest_cfg in cfg.destinations:
        dest_code = dest_cfg["code"].upper()
        max_price = float(dest_cfg["max_price_usd"])
        log.info("=== %s -> %s (ceiling $%.0f) ===", cfg.origin, dest_code, max_price)

        dest_air = resolve_airport(dest_code, cfg.rapidapi_key, airport_cache)
        if not dest_air:
            log.warning("could not resolve %s, skipping", dest_code)
            continue

        candidates: list[FlightDeal] = []
        for depart, return_ in sample_date_pairs(cfg, seed_salt=dest_code):
            itins = search_flights(cfg, origin_air, dest_air, depart, return_)
            if itins is None:
                continue
            deals = normalize(itins, cfg.origin, dest_code, depart, return_, cfg.cabin_class)
            total_seen += len(deals)
            for deal in deals:
                if deal.price_usd > max_price:
                    continue
                if state.get(deal.fingerprint()):
                    log.info("  skip already-alerted %s $%.0f", deal.depart_date, deal.price_usd)
                    continue
                candidates.append(deal)
            time.sleep(1.0)  # gentle pacing for RapidAPI

        if not candidates:
            log.info("  no candidates under ceiling")
            continue

        # Score only the cheapest few per destination to control LLM cost
        candidates.sort(key=lambda d: d.price_usd)
        for deal in candidates[:3]:
            try:
                score, reasoning = score_deal(client, deal, max_price)
            except Exception as e:
                log.warning("scoring failed for %s %s: %s", dest_code, deal.depart_date, e)
                continue
            log.info("  scored %s $%.0f -> %d/10 (%s)",
                     deal.depart_date, deal.price_usd, score, reasoning[:60])
            if score >= cfg.min_score:
                msg = format_alert(deal, score, reasoning)
                if dry_run:
                    log.info("  [DRY-RUN] would alert:\n%s", msg)
                else:
                    send_telegram(cfg, msg)
                state[deal.fingerprint()] = datetime.now(timezone.utc).isoformat()
                total_alerted += 1

    save_state(state)
    log.info("done. %d itineraries seen, %d alerts sent.", total_seen, total_alerted)


def force_alert(cfg: Config) -> None:
    fake = FlightDeal(
        origin=cfg.origin,
        destination="TST",
        depart_date="2099-01-01",
        return_date="2099-01-10",
        price_usd=99.0,
        total_duration_minutes=600,
        stops_outbound=0,
        stops_inbound=0,
        layover_minutes=0,
        carriers=["Test Air"],
        deep_link="https://example.com",
    )
    send_telegram(cfg, format_alert(fake, 10, "End-to-end test ping from flight_agent.py."))
    log.info("synthetic alert sent.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="run normally but skip Telegram and skip state writes")
    parser.add_argument("--force-alert", action="store_true",
                        help="send a synthetic Telegram message and exit (E2E test)")
    args = parser.parse_args()

    cfg = load_config()
    if args.force_alert:
        force_alert(cfg)
        return
    run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

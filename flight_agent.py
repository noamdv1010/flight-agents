"""Autonomous flight-deal hunter.

Pipeline (per run):
    config.yaml + secrets
        -> fast-flights (Google Flights, no API key) per destination & date sample
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
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests
import yaml
from anthropic import Anthropic
from fast_flights import FlightData, Passengers, get_flights

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
STATE_PATH = SCRIPT_DIR / "state.json"

# fast-flights fetch mode:
#   "common"         - direct HTTP, parses Google's protobuf response (default, fastest)
#   "fallback"       - tries common first, falls back to try.playwright.tech if empty.
#                      That fallback now requires a paid token, so prefer "common".
#   "force-fallback" - skips common entirely; needs a try.playwright.tech token.
#   "local"          - runs Playwright locally; reliable but heavy.
FETCH_MODE = os.environ.get("FAST_FLIGHTS_MODE", "common").strip()

# Google Flights returns prices in the locale's currency based on the source IP.
# These are approximate USD conversion rates for filtering against the user's
# USD-denominated ceilings. ±5% is fine for mistake-fare triage; users who care
# about exactness can override via the `fx_rates` block in config.yaml.
FX_TO_USD_DEFAULTS: dict[str, float] = {
    "$": 1.0,
    "US$": 1.0,
    "USD": 1.0,
    "€": 1.07,
    "EUR": 1.07,
    "£": 1.27,
    "GBP": 1.27,
    "₪": 0.27,
    "ILS": 0.27,
    "¥": 0.0067,
    "JPY": 0.0067,
    "₩": 0.00073,
    "KRW": 0.00073,
    "₹": 0.012,
    "INR": 0.012,
    "C$": 0.73,
    "CAD": 0.73,
    "A$": 0.66,
    "AUD": 0.66,
}

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
    fx_rates: dict[str, float] = field(default_factory=dict)

    # Secrets
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

    fx_rates = dict(FX_TO_USD_DEFAULTS)
    fx_rates.update(raw.get("fx_rates", {}) or {})

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
        fx_rates=fx_rates,
        anthropic_key=need_env("ANTHROPIC_API_KEY"),
        telegram_token=need_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=need_env("TELEGRAM_CHAT_ID"),
    )


def load_state() -> dict[str, dict[str, Any]]:
    """Load state.json. Migrates legacy string-only entries into the rich
    dict-shape used by the relative-scoring logic.

    New entry shape:
        {
          "alerted_at": "<ISO timestamp>",
          "route": "TLV-ATH",
          "price_usd": 220.0,
          "score": 8,
          "depart_date": "2026-08-15",
          "return_date": "2026-08-22"
        }
    """
    if not STATE_PATH.exists():
        return {}
    try:
        raw = json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        log.warning("state.json corrupt, starting fresh")
        return {}
    migrated: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            # legacy entry: just the timestamp, no route/price/score history.
            migrated[k] = {"alerted_at": v}
        elif isinstance(v, dict) and "alerted_at" in v:
            migrated[k] = v
        # silently drop anything malformed
    return migrated


def save_state(state: dict[str, dict[str, Any]]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False))


def prune_state(state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=STATE_TTL_DAYS)
    out: dict[str, dict[str, Any]] = {}
    for k, v in state.items():
        ts = v.get("alerted_at")
        if not ts:
            continue
        try:
            if datetime.fromisoformat(ts) > cutoff:
                out[k] = v
        except ValueError:
            continue
    return out


def recent_alerts_for_route(
    state: dict[str, dict[str, Any]],
    origin: str,
    destination: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return up to `limit` recent alerts for this exact route, cheapest first.

    Used to give the LLM a baseline so it can score new deals *relatively*
    against what has already been alerted on, instead of in isolation.
    """
    route = f"{origin}-{destination}"
    matches = [
        v for v in state.values()
        if v.get("route") == route and "price_usd" in v
    ]
    matches.sort(key=lambda x: float(x.get("price_usd", 9e9)))
    return matches[:limit]


# ---------------------------------------------------------------------------
# Google Flights via fast-flights (no API key, no quota — but ToS-grey and
# subject to break when Google rotates their internal protobuf)
# ---------------------------------------------------------------------------


_DURATION_RE_HR = re.compile(r"(\d+)\s*hr", re.IGNORECASE)
_DURATION_RE_MIN = re.compile(r"(\d+)\s*min", re.IGNORECASE)


def parse_price(raw: str, fx_rates: dict[str, float]) -> float:
    """Convert a fast-flights price string (locale-dependent) into USD.

    Examples: '$249' -> 249.0, '₪1868' -> ~504, '€350' -> ~374.5
    Returns 0.0 if the string can't be parsed.
    """
    if not raw:
        return 0.0
    s = raw.strip()
    # Longest-match so 'US$' wins over '$'.
    symbol = next(
        (k for k in sorted(fx_rates, key=len, reverse=True) if k in s),
        None,
    )
    digits = re.sub(r"[^\d.]", "", s)
    if not digits:
        return 0.0
    try:
        amount = float(digits)
    except ValueError:
        return 0.0
    rate = fx_rates.get(symbol, 1.0) if symbol else 1.0
    return amount * rate


def parse_duration_minutes(raw: str) -> int:
    """Parse human strings like '8 hr 15 min' or '45 min' into minutes."""
    if not raw:
        return 0
    h = _DURATION_RE_HR.search(raw)
    m = _DURATION_RE_MIN.search(raw)
    return (int(h.group(1)) if h else 0) * 60 + (int(m.group(1)) if m else 0)


def _google_flights_deep_link(origin: str, destination: str, depart: str, return_: str) -> str:
    """Build a Google Flights deep-link that opens directly to this itinerary.

    Uses the format Google Flights itself uses when sharing a search:
      https://www.google.com/travel/flights?q=Flights+to+{DST}+from+{ORI}+on+{DATE}+through+{DATE}
    """
    q = f"Flights to {destination} from {origin} on {depart} through {return_}"
    return f"https://www.google.com/travel/flights?q={quote_plus(q)}"


def _call_fast_flights(cfg: Config, origin: str, destination: str, depart: str, return_: str):
    return get_flights(
        flight_data=[
            FlightData(date=depart, from_airport=origin, to_airport=destination),
            FlightData(date=return_, from_airport=destination, to_airport=origin),
        ],
        trip="round-trip",
        seat=cfg.cabin_class,
        passengers=Passengers(adults=cfg.adults),
        fetch_mode=FETCH_MODE,
    )


def search_flights(
    cfg: Config,
    origin: str,
    destination: str,
    depart: str,
    return_: str,
) -> list[FlightDeal]:
    """Call fast-flights for one round-trip and normalize directly into FlightDeal.

    fast-flights returns a list of outbound options, each carrying the cheapest
    full round-trip price available with that outbound. The response has no
    inbound or layover detail, so we set those fields to sentinel values (-1)
    and the scoring prompt tells Claude to treat them as 'unknown'.

    Google occasionally returns an unrendered loading-page (0 flights). We
    retry once after a short pause; if it still comes back empty, we accept it.
    """
    result = None
    for attempt in (1, 2):
        try:
            result = _call_fast_flights(cfg, origin, destination, depart, return_)
            if result.flights:
                break
            log.info("empty result %s->%s %s/%s (attempt %d)",
                     origin, destination, depart, return_, attempt)
        except Exception as e:
            # Truncate noisy HTML error pages so the log stays readable.
            msg = str(e).splitlines()[0][:200]
            log.warning("fast-flights %s->%s %s/%s attempt %d failed: %s",
                        origin, destination, depart, return_, attempt, msg)
        if attempt == 1:
            time.sleep(6.0)

    if not result or not result.flights:
        return []

    deals: list[FlightDeal] = []
    deep_link = _google_flights_deep_link(origin, destination, depart, return_)
    for f in result.flights or []:
        try:
            price_usd = parse_price(f.price, cfg.fx_rates)
            if price_usd <= 0:
                continue
            outbound_min = parse_duration_minutes(f.duration)
            stops = int(f.stops or 0)
            deals.append(FlightDeal(
                origin=origin,
                destination=destination,
                depart_date=depart,
                return_date=return_,
                price_usd=price_usd,
                # Only outbound duration is available from fast-flights; the
                # scoring prompt mentions this so Claude doesn't double-count.
                total_duration_minutes=outbound_min,
                stops_outbound=stops,
                stops_inbound=-1,        # sentinel: unknown
                layover_minutes=-1,      # sentinel: unknown
                carriers=[f.name] if f.name else [],
                deep_link=deep_link,
                cabin_class=cfg.cabin_class,
            ))
        except (AttributeError, TypeError, ValueError) as e:
            log.debug("skipping malformed flight: %s", e)
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
        if (d2 - d1).days < 2:          # hard floor: must be at least 2 nights
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


SCORING_SYSTEM = """אתה מעריך עסקאות טיסה מדוקדק. אתה מדרג כרטיסי טיסה הלוך-חזור בסקלה של 1-10:

  10 = טעות מחיר או עסקה יוצאת דופן — להזמין תוך שעה
   9 = עסקה מצוינת, נמוכה משמעותית מהרגיל, לוח זמנים נוח
   8 = עסקה טובה בבירור שכדאי להתריע עליה
   7 = מחיר סביר אך עם חיכוך (עצירה ארוכה, שעות מסורבלות, חברה חלשה)
   1-6 = לא שווה להפריע למשתמש

אתה שוקל את הגורמים הבאים:
  * מחיר מול בסיס סביר לנתיב ולעונה — המשקל הגדול ביותר
  * משך הטיסה היוצאת (מקור הנתונים נותן רק כיוון אחד)
  * מספר עצירות בטיסה היוצאת — טיסה ישירה היא יתרון חזק
  * מוניטין חברת התעופה — חברות ראשיות ו-LCC מכובדות בסדר; חברות לא מוכרות מורידות את הציון
  * אורך הנסיעה המתאים לנוסע אמיתי

חלק מהשדות עשויים להיות "לא ידוע". אל תעניש על שדות חסרים; הסתמך יותר על השדות הקיימים, בעיקר מחיר ועצירות.

** דירוג יחסי להיסטוריה — חשוב מאוד: **
אם מוצגת לך "היסטוריית התראות אחרונות לנתיב הזה", השתמש בה כקו בסיס:
  * דיל חדש באותו טווח מחיר (±5%) של דיל שכבר התרענו עליו → מקסימום 7
  * דיל יקר יותר מהדיל הזול ביותר בהיסטוריה → מקסימום 7
  * דיל זול ב-10%-20% מהזול ביותר בהיסטוריה → ראוי לציון 8
  * דיל זול ב-20%-35% מהזול ביותר בהיסטוריה → ציון 9
  * דיל זול ב-35%+ מהזול ביותר בהיסטוריה או טעות מחיר ברורה → ציון 10
המטרה: לא להציף את המשתמש שוב ושוב באותו דיל. אם כבר התרענו על אתונה ב-$220, דיל חדש באתונה ב-$215 לא ראוי להתראה — רק מחיר נמוך משמעותית מצדיק זאת.

היה קפדן: רוב המחירים לא אמורים לקבל ציון 8+. שמור ציונים גבוהים לעסקאות אמיתיות.
החזר תשובה אך ורק דרך הכלי `record_deal_score`. כתוב את שדה reasoning בעברית בלבד, עד 200 תווים."""


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
                "description": "משפט אחד בעברית המסביר את הגורמים המרכזיים לדירוג.",
            },
        },
        "required": ["score", "reasoning"],
    },
}


def score_deal(
    client: Anthropic,
    deal: FlightDeal,
    max_price_for_route: float,
    recent_alerts: list[dict[str, Any]] | None = None,
) -> tuple[int, str]:
    def fmt_min(m: int) -> str:
        return "unknown" if m < 0 else f"{m // 60}h {m % 60}m"

    def fmt_stops(s: int) -> str:
        return "unknown" if s < 0 else str(s)

    # Render history so the LLM scores this deal relative to past alerts, not in isolation.
    history_block = ""
    if recent_alerts:
        lines = ["", "היסטוריית התראות אחרונות לנתיב הזה (14 ימים אחרונים, מסודר מהזול ליקר):"]
        for a in recent_alerts:
            price = a.get("price_usd", 0)
            sc = a.get("score", "?")
            dep = a.get("depart_date", "?")
            ret = a.get("return_date", "?")
            lines.append(f"  - ${price:.0f} (ציון {sc}/10), {dep} → {ret}")
        cheapest = min(float(a.get("price_usd", 9e9)) for a in recent_alerts)
        lines.append(
            f"\nהמחיר הזול ביותר שכבר התרענו עליו: ${cheapest:.0f}. "
            f"דרג את הדיל החדש ביחס לקו הבסיס הזה לפי הכללים בסיסטם פרומפט."
        )
        history_block = "\n".join(lines) + "\n"

    user_msg = (
        f"Route: {deal.origin} <-> {deal.destination}\n"
        f"Dates: {deal.depart_date} to {deal.return_date} ({deal.trip_length()} nights)\n"
        f"Price: ${deal.price_usd:.0f} USD (user's alert ceiling for this route: ${max_price_for_route:.0f})\n"
        f"Cabin: {deal.cabin_class}\n"
        f"Outbound flight duration: {fmt_min(deal.total_duration_minutes)}\n"
        f"Stops: outbound={fmt_stops(deal.stops_outbound)}, inbound={fmt_stops(deal.stops_inbound)}\n"
        f"Longest single layover: {fmt_min(deal.layover_minutes)}\n"
        f"Carriers: {', '.join(deal.carriers) or 'unknown'}\n"
        f"{history_block}\n"
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
    """Build an HTML-formatted Telegram message in Hebrew. All dynamic strings are escaped."""
    def fmt_dur(m: int) -> str:
        return "לא ידוע" if m < 0 else f"{m // 60}ש׳ {m % 60:02d}ד׳"

    if deal.stops_outbound < 0:
        stops_str = "לא ידוע"
    elif deal.stops_outbound == 0:
        stops_str = "ישירה ✅"
    else:
        stops_str = f"{deal.stops_outbound} עצירות"

    carriers = ", ".join(deal.carriers[:3]) or "לא ידוע"

    # Score badge: 10 = 🔥, 9 = ⭐⭐, 8 = ⭐
    if score == 10:
        badge = "🔥🔥🔥"
    elif score == 9:
        badge = "⭐⭐"
    else:
        badge = "⭐"

    e = html.escape  # all user/api-controlled strings flow through this
    return (
        f"✈️ <b>דיל מטורף {badge} — {score}/10</b>\n"
        f"📍 {e(deal.origin)} ← → {e(deal.destination)}\n\n"
        f"💰 <b>מחיר הלוך-חזור:</b> <b>${deal.price_usd:.0f}</b>\n"
        f"📅 <b>יציאה:</b> {e(deal.depart_date)}\n"
        f"🔄 <b>חזרה:</b> {e(deal.return_date)} ({deal.trip_length()} לילות)\n"
        f"⏱ <b>משך טיסה:</b> {fmt_dur(deal.total_duration_minutes)} | {stops_str}\n"
        f"🛫 <b>חברת תעופה:</b> {e(carriers)}\n\n"
        f"💬 <i>{e(reasoning)}</i>\n\n"
        f'🔗 <a href="{e(deal.deep_link, quote=True)}">פתח בגוגל פלייטס</a>\n\n'
        f"⚠️ <i>שים לב: המחיר הינו לכרטיס בסיסי (תיק גב/טרולי). "
        f"בחברות לואו-קוסט תיתכן תוספת עבור מזוודה גדולה.</i>"
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(cfg: Config, dry_run: bool) -> None:
    client = Anthropic(api_key=cfg.anthropic_key)
    state = prune_state(load_state())

    total_seen = 0
    total_alerted = 0

    for dest_cfg in cfg.destinations:
        dest_code = dest_cfg["code"].upper()
        max_price = float(dest_cfg["max_price_usd"])
        log.info("=== %s -> %s (ceiling $%.0f) ===", cfg.origin, dest_code, max_price)

        candidates: list[FlightDeal] = []
        for depart, return_ in sample_date_pairs(cfg, seed_salt=dest_code):
            deals = search_flights(cfg, cfg.origin, dest_code, depart, return_)
            total_seen += len(deals)
            for deal in deals:
                if deal.price_usd > max_price:
                    continue
                if state.get(deal.fingerprint()):
                    log.info("  skip already-alerted %s $%.0f", deal.depart_date, deal.price_usd)
                    continue
                candidates.append(deal)
            # Pacing — Google can rate-limit aggressive scraping.
            time.sleep(2.0)

        if not candidates:
            log.info("  no candidates under ceiling")
            continue

        # Score only the cheapest few per destination to control LLM cost.
        # Pull the recent-alert history for this route ONCE and feed it into
        # every scoring call so the LLM can compare against the existing baseline.
        history = recent_alerts_for_route(state, cfg.origin, dest_code)
        if history:
            log.info("  history for %s: %d recent alerts (cheapest $%.0f)",
                     dest_code, len(history), min(a["price_usd"] for a in history))

        candidates.sort(key=lambda d: d.price_usd)
        for deal in candidates[:3]:
            try:
                score, reasoning = score_deal(client, deal, max_price, recent_alerts=history)
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
                state[deal.fingerprint()] = {
                    "alerted_at": datetime.now(timezone.utc).isoformat(),
                    "route": f"{deal.origin}-{deal.destination}",
                    "price_usd": round(deal.price_usd, 2),
                    "score": score,
                    "depart_date": deal.depart_date,
                    "return_date": deal.return_date,
                }
                # Future deals in THIS same run for the same destination should
                # also see this alert as part of their baseline.
                history = recent_alerts_for_route(state, cfg.origin, dest_code)
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
        stops_inbound=-1,
        layover_minutes=-1,
        carriers=["Test Air"],
        deep_link="https://example.com",
    )
    send_telegram(cfg, format_alert(fake, 10, "בדיקת קצה לקצה — הודעת טסט מ-flight_agent.py."))
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

# Flight Agent

Autonomous mistake-fare hunter. Runs on GitHub Actions every 4 hours, scans
your watchlist via Google Flights (via the `fast-flights` library — no API key,
no quota), has Claude Haiku 4.5 score each candidate, and pings Telegram when a
deal scores >= 8/10.

## Cost

- GitHub Actions: free (job runs ~3 min, well under the 2,000 min/mo private quota)
- Telegram: free
- Google Flights: free (no API key — `fast-flights` calls Google's internal endpoints)
- Anthropic Haiku 4.5: ~$0.50–$2/mo at default settings

## Heads up about the data source

`fast-flights` scrapes Google Flights' internal protobuf API. It's:
- **Free and quota-less** — main reason for the choice.
- **Sits in ToS-grey territory.** Fine for personal-scale use; don't deploy this for a business.
- **Breaks every few months** when Google rotates their schema. When that happens, `pip install --upgrade fast-flights` usually fixes it.
- **More likely to be rate-limited from cloud IPs** (like GitHub Actions). The default `FAST_FLIGHTS_MODE=fallback` routes through the library's hosted proxy to help.

If reliability matters more than zero-cost, swap to **Travelpayouts** (1000 req/day free, real API) or **Amadeus Self-Service** (2000/mo free, official airline data).

## 1. Get your API keys

### Anthropic key
1. https://console.anthropic.com → Settings → API Keys → Create Key

### Telegram bot
See the next section.

## 2. Create the Telegram bot (step by step)

1. **Open Telegram** and search for `@BotFather` (verified blue check).
2. Send `/start`, then `/newbot`.
3. BotFather asks for a **display name** — anything you like (e.g. `My Flight Hunter`).
4. Then a **username** — must end in `bot` (e.g. `noam_flight_hunter_bot`).
5. BotFather replies with a token like `7841234567:AAH...xyz`. **That's your `TELEGRAM_BOT_TOKEN`.** Save it.
6. **Important:** the bot can only message you after *you* message it first. Open your new bot in Telegram and send any message (e.g. `hi`).
7. Get your chat ID. In a terminal:
   ```bash
   curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
   ```
   In the JSON response find `result[0].message.chat.id` — a number like `123456789`. **That's your `TELEGRAM_CHAT_ID`.**

Test end-to-end after setting the env vars:
```bash
python flight_agent.py --force-alert
```
A formatted message should arrive in Telegram within a couple of seconds.

## 3. Local smoke test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in the three keys
set -a; source .env; set +a
python flight_agent.py --dry-run   # scans real data, prints what would alert, no Telegram, no state writes
```

## 4. Deploy to GitHub Actions

1. Create a **private** GitHub repo (state will be committed back to it, so private is recommended).
2. Push this directory:
   ```bash
   git init && git add . && git commit -m "initial"
   git branch -M main
   git remote add origin git@github.com:<you>/flight-agents.git
   git push -u origin main
   ```
3. In the repo: **Settings → Secrets and variables → Actions → New repository secret**. Add all three:
   - `ANTHROPIC_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
4. **Actions tab → flight-scan → Run workflow** to trigger the first run manually. Watch the log.
5. After that the cron (`0 */4 * * *`) runs it automatically.

## 5. Tuning

Edit `config.yaml`:
- `destinations` / `max_price_usd`: your watchlist and the price threshold that marks a "deal" for that route
- `min_score`: raise to 9 for fewer/better alerts, lower to 7 for more noise
- `samples_per_destination`: each sample = 1 fast-flights call. Pacing is ~2s between calls.
- `date_window_days`: how far out to scan
- `trip_length_days`: e.g. `[3, 5]` for weekend trips, `[10, 21]` for longer holidays
- `fx_rates` (optional): override the built-in currency conversion table if Google Flights returns prices in a currency you want to convert more accurately

To change cadence, edit the cron in `.github/workflows/flight-scan.yml`.

## How alerts are deduped

Each alerted itinerary is hashed (origin + destination + dates + price bucketed
to nearest $10) and stored in `state.json` for 14 days. So you won't get
re-pinged about the same deal on every run — but a real price drop of more than
~$10 will re-trigger.

## When fast-flights breaks

If the workflow starts logging `fast-flights ... failed: ...` errors:

1. `pip install --upgrade fast-flights` locally and check if a new version is available.
2. Bump `fast-flights>=X.Y` in `requirements.txt` and push.
3. If the library hasn't been updated, try `FAST_FLIGHTS_MODE=common` (direct requests) or `local` (Playwright, heavier).
4. Worst case: swap data sources. The `search_flights()` function in `flight_agent.py` is the only place you'd need to change.

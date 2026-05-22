# Flight Agent

Autonomous mistake-fare hunter. Runs on GitHub Actions every 4 hours, scans
your watchlist via Skyscanner (sky-scrapper on RapidAPI), has Claude Haiku 4.5
score each candidate, and pings Telegram when a deal scores >= 8/10.

## Cost

- GitHub Actions: free (job runs ~2 min, well under the 2,000 min/mo private quota)
- Telegram: free
- RapidAPI (`sky-scrapper`): free tier is ~50 requests/mo; BASIC tier $10/mo gives 500. Tune `samples_per_destination` and cron frequency to fit.
- Anthropic Haiku 4.5: ~$0.50–$2/mo at default settings

## 1. Get your API keys

### RapidAPI key
1. Sign up at https://rapidapi.com
2. Subscribe to the **Sky Scrapper** API by `apiheya` (search "sky-scrapper")
3. Copy the `X-RapidAPI-Key` from the dashboard

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
cp .env.example .env        # fill in all four keys
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
3. In the repo: **Settings → Secrets and variables → Actions → New repository secret**. Add all four:
   - `RAPIDAPI_KEY`
   - `ANTHROPIC_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
4. **Actions tab → flight-scan → Run workflow** to trigger the first run manually. Watch the log.
5. After that the cron (`0 */4 * * *`) runs it automatically.

## 5. Tuning

Edit `config.yaml`:
- `destinations` / `max_price_usd`: your watchlist and the price threshold that marks a "deal" for that route
- `min_score`: raise to 9 for fewer/better alerts, lower to 7 for more noise
- `samples_per_destination`: each sample = 1 RapidAPI call. Lower this if you're on the free tier
- `date_window_days`: how far out to scan
- `trip_length_days`: e.g. `[3, 5]` for weekend trips, `[10, 21]` for longer holidays

To change cadence, edit the cron in `.github/workflows/flight-scan.yml`.

## How alerts are deduped

Each alerted itinerary is hashed (origin + destination + dates + price bucketed
to nearest $10) and stored in `state.json` for 14 days. So you won't get
re-pinged about the same deal on every run — but a real price drop of more than
~$10 will re-trigger.

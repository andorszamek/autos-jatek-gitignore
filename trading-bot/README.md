# AI Trading Bot V2

Automata részvény-kereskedő bot Pythonban, LightGBM ML modellel.
**Alapértelmezésben CSAK paper trading.** Valódi pénzes kereskedés nem engedélyezett,
amíg nem érted és nem vállalod a kockázatot (lásd lent).

> **KOCKÁZATI FIGYELMEZTETÉS:** Ez a szoftver kísérletezési célt szolgál.
> Nem garantál nyereséget. Tőzsdei kereskedéssel elveszítheted a befektetett
> tőke egy részét vagy egészét. Paper módban tesztelj hónapokig, mielőtt
> bármilyen valódi pénzt kockáztatsz.

---

## Gyors indítás (Mac — tréning + fejlesztés)

### 1. Előfeltételek

- Python 3.11+
- `uv` (ajánlott) vagy `pip`

### 2. Környezet felállítása

```bash
# uv-vel (ajánlott)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# vagy hagyományos pip-pel
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. `.env` kitöltése

```bash
cp .env.example .env
```

Szerkeszd a `.env`-t:

```
ALPACA_API_KEY=<Alpaca paper API kulcsod>
ALPACA_SECRET_KEY=<Alpaca paper secret kulcsod>
ALPACA_PAPER=true
LIVE_TRADING=false
```

Alpaca paper kulcsot itt kaphatsz: https://app.alpaca.markets → Paper account → API Keys

### 4. Kapcsolat tesztelése

```bash
python scripts/smoke_test_account.py
```

### 5. Teljes workflow (Mac)

```bash
# Adatok letöltése
python scripts/download_data.py

# Modell tréning
python scripts/train.py

# Backtest futtatása (költségekkel)
python scripts/run_backtest.py

# Paper bot egyszeri futtatása
python scripts/run_paper.py
```

---

## Raspberry Pi telepítés (futtatás)

A Pi **NEM tréningel** — csak a betanított modellt futtatja.

### 1. Szükséges fájlok átmásolása

```bash
# Mac-en: modell átküldése Pi-re
scp models/latest.lgb pi@<PI_IP>:~/trading-bot/models/latest.lgb
scp -r trading-bot/ pi@<PI_IP>:~/
```

### 2. Pi-n: csak runtime függőségek

```bash
pip install alpaca-py pandas pyarrow lightgbm python-dotenv pyyaml
```

### 3. Systemd service

Lásd `systemd/trading-bot.service` — napi egyszeri futtatás tőzsde zárás után.

```bash
sudo cp systemd/trading-bot.service /etc/systemd/system/
sudo systemctl enable trading-bot.timer
sudo systemctl start trading-bot.timer
```

---

## Raspberry Pi Setup (English)

The Pi is a **run-only** node — it never trains the model.
Training always happens on a Mac/desktop with the full ML stack.

### Step-by-step

**1. Clone the repo on the Pi**

```bash
git clone <repo-url> ~/trading-bot
cd ~/trading-bot
```

**2. Create a virtual environment and install runtime dependencies only**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install alpaca-py pandas pyarrow lightgbm python-dotenv pyyaml yfinance requests
```

> `lightgbm` is needed for inference (loading and running the model).
> Heavy training dependencies (`scikit-learn`, `vectorbt`, etc.) are NOT needed on the Pi.

**3. Copy `.env` and the trained model from your Mac**

```bash
# On your Mac:
scp .env pi@<PI_IP>:~/trading-bot/.env
scp models/latest.lgb pi@<PI_IP>:~/trading-bot/models/latest.lgb
```

**4. Set up systemd service and timer**

```bash
sudo cp systemd/trading-bot.service /etc/systemd/system/
sudo cp systemd/trading-bot.timer   /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable trading-bot.timer
sudo systemctl start  trading-bot.timer
```

Verify the timer is active:

```bash
systemctl status trading-bot.timer
systemctl list-timers trading-bot.timer
```

**5. Updating the model**

Retrain on your Mac, then push the new model to the Pi:

```bash
# On your Mac (after running scripts/train.py):
scp models/latest.lgb pi@<PI_IP>:~/trading-bot/models/latest.lgb
```

The next timer cycle will automatically use the new model.

### Health check log location

After each run, the Pi writes a health status file:

```
~/trading-bot/logs/health.json
```

Example contents:

```json
{
  "last_run": "2024-01-15T22:00:01+00:00",
  "last_success": "2024-01-15T22:00:01+00:00",
  "last_error": null
}
```

Tail the journal for live logs:

```bash
journalctl -u trading-bot.service -f
```

Or inspect structured JSON logs:

```bash
tail -f ~/trading-bot/logs/trading_$(date +%Y%m%d).log | python3 -m json.tool
```

### Note: Pi never runs training

- **Never** run `scripts/train.py` on the Pi — it requires heavy dependencies not installed.
- The Pi only runs `scripts/run_paper.py` via the systemd timer.
- Always train on a machine with the full `requirements.txt` stack installed.

---

## Élő kereskedés (NE csináld, amíg hónapokig nem paperezik)

Élő kereskedés aktiválásához **mindkét** feltétel szükséges egyszerre:

1. `.env`-ben: `LIVE_TRADING=true`
2. CLI flag: `--i-understand-the-risk`

```bash
python scripts/run_paper.py --i-understand-the-risk
```

**Ha valamelyik hiányzik, a bot automatikusan paper módra vált vissza.**
Ez a logika minden megbízás előtt ellenőrzött — nem csak induláskor.

---

## Repo struktúra

```
trading-bot/
├── src/           # Fő modulok
├── scripts/       # CLI szkriptek
├── tests/         # Pytest tesztek
├── config.yaml    # Konfiguráció (szerkeszthető)
├── .env.example   # Környezeti változók sablona
└── requirements.txt
```

## Tesztek futtatása

```bash
cd trading-bot
pytest tests/ -v
```

## Amit NE csinálj

- Ne commitolj `.env` fájlt (gitignore-olt)
- Ne bízz bele blindán — a modell nem lát a jövőbe
- Ne légy éles módban, amíg a paper backtest és live paper eredmények nem meggyőzőek
- Ne tárolj API kulcsot a kódban

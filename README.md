
# AI Trader – Dual EMA (Streamlit) v3.2

Nowości:
- ✅ **„Zachowaj sygnały modelu (No‑Regret tylko na equity)”** — No‑Regret nie nadpisuje sygnałów, jedynie equity.
- ✅ **Dzisiejsza rekomendacja** (BUY/ADD/SELL/HOLD/ACCUMULATE) wyliczana na bazie najlepszych parametrów.
- ✅ Stooq fetch, Momentum bias, dual‑EMA validation i porównanie z Buy&Hold.

## Start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_streamlit.py
```

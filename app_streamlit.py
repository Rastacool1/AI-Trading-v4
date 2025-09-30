
import io, sys, requests
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
from engine import load_price_csv, adaptive_stats, random_optimize, bench_roi, simulate_today_recommendation

st.set_page_config(page_title="AI Trader – Dual EMA", layout="wide")
st.title("AI Trader – Dual EMA (EMA15/40/100) — v3.2")

@st.cache_data(show_spinner=False)
def fetch_stooq(symbol: str, interval: str="d") -> pd.DataFrame:
    url = f"https://stooq.pl/q/d/l/?s={symbol.lower()}&i={interval}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

with st.sidebar:
    st.header("Źródło danych")
    tab = st.radio("Wybierz źródło", ["Upload CSV", "Stooq.pl"], index=0)
    if tab == "Upload CSV":
        file = st.file_uploader("Wgraj CSV (stooq/yahoo)", type=["csv"])
        st.caption("CSV musi zawierać kolumny Data/Date oraz Zamkniecie/Close (Adj Close też działa).")
        df_raw = None
        if file is not None:
            df_raw = pd.read_csv(file)
    else:
        symbol = st.text_input("Symbol Stooq (np. btcusd, es.c, wig20)", "wig20")
        interval = st.selectbox("Interwał", ["d","w","m"], index=0)
        fetch = st.button("Pobierz z Stooq")
        df_raw = None
        if fetch and symbol:
            try:
                df_raw = fetch_stooq(symbol, interval)
                st.success(f"Pobrano {len(df_raw)} wierszy dla {symbol} ({interval}).")
            except Exception as e:
                st.error(f"Nie udało się pobrać danych: {e}")

    st.header("Backtest")
    start = st.text_input("Start (YYYY-MM-DD)", value="2023-01-01")
    short_mode = st.selectbox("Krótka EMA (walidacja SELL)", ["40","15","dyn"], index=0,
                              help="15/40 lub adaptacyjna dyn = EMA(L/2)")
    samples = st.slider("Liczba prób (losowe)", 50, 500, 150, step=10)
    seed    = st.number_input("Seed", value=11, step=1)

    st.write("---")
    run_btn = st.button("Optymalizuj i generuj sygnały")
    run_mom = st.button("Optymalizuj (300) + Momentum bias")

    st.write("---")
    st.subheader("Tryb No‑Regret (Kup i trzymaj po słabszym okresie)")
    no_regret = st.checkbox("Włącz No‑Regret fallback", value=False,
                            help="Jeśli model przez X miesięcy z rzędu jest słabszy niż zwykłe Kup‑i‑Trzymaj, aplikacja przełącza equity na B&H, żeby nie przegrać z pasywnym podejściem.")
    months = st.slider("Liczba miesięcy słabszych od B&H (z rzędu)", 2, 12, 3, step=1)
    keep_model_signals = st.checkbox("Zachowaj sygnały modelu (No‑Regret tylko na equity)", value=True,
                                     help="Gdy włączone: sygnały BUY/SELL/ADD pozostają z modelu. No‑Regret modyfikuje tylko wykres equity.")
    st.caption("Prosto mówiąc: to „bez żalu” – gdy algorytm męczy się długo, nie kombinujemy, tylko trzymamy.")

    st.write("---")
    if st.button("Wyczyść cache"):
        st.cache_data.clear()
        st.success("Cache wyczyszczony.")

if not run_btn and not run_mom:
    st.info("Wgraj lub pobierz dane, ustaw parametry i kliknij **Optymalizuj**.")
    st.stop()

if df_raw is None:
    st.error("Brak danych – wgraj/pobierz CSV.")
    st.stop()

# Parse with engine loader for column flexibility
try:
    buf = io.StringIO(); df_raw.to_csv(buf, index=False); buf.seek(0)
    df = load_price_csv(buf)
except Exception as e:
    st.exception(e); st.stop()

bias = "momentum" if run_mom else "none"
eff_samples = 300 if run_mom else int(samples)

with st.spinner("Liczenie..."):
    info, sig, trd = random_optimize(df, start=start, n_samples=eff_samples, seed=int(seed), short_ema_mode=short_mode, bias=bias)

# Dzisiejsza rekomendacja (z parametrów best_params)
today_rec = None
if info.get("best_params") is not None:
    try:
        today_rec = simulate_today_recommendation(df, start=start, params=info["best_params"], short_ema_mode=short_mode)
    except Exception as e:
        today_rec = {"action":"HOLD","note":f"Błąd rekomendacji: {e}", "price": float(df['Zamkniecie'].iloc[-1])}
else:
    today_rec = {"action":"HOLD","note":"Brak wybranych parametrów (fallback B&H?).","price": float(df['Zamkniecie'].iloc[-1])}

# === Equity curves (clean) + No‑Regret ===
idx = df.loc[pd.to_datetime(start):].index
curve = pd.Series(100.0, index=idx, name="Model")
if not trd.empty:
    cum = 100.0
    for _, t in trd.iterrows():
        d = pd.to_datetime(t['exit_date'])
        if d in curve.index:
            cum += float(t['pnl_usd'])
            curve.loc[d:] = cum
bh = 100.0 * (df.loc[pd.to_datetime(start):,'Zamkniecie'] / df.loc[pd.to_datetime(start):,'Zamkniecie'].iloc[0])
bh.name = "Buy&Hold"
eq = pd.concat([curve, bh], axis=1).ffill()

switch_date = None
sig_adj = sig.copy()
if no_regret:
    m = eq.resample('M').last().dropna()
    under = (m['Model'] < m['Buy&Hold']).astype(int)
    run = (under.groupby((under != under.shift()).cumsum()).cumsum()) * under
    idx_under = run.index[run >= months]
    if len(idx_under)>0:
        switch_month_end = idx_under[0]
        after = eq.index[eq.index > switch_month_end]
        if len(after)>0:
            switch_date = after[0]
            eq.loc[eq.index >= switch_date, "Model"] = eq.loc[eq.index >= switch_date, "Buy&Hold"]
            if not keep_model_signals:
                last_date = df.index[-1]
                px_buy = float(df.loc[switch_date, 'Zamkniecie']) if switch_date in df.index else float(df.iloc[-1]['Zamkniecie'])
                px_sell= float(df.iloc[-1]['Zamkniecie'])
                sig_adj = pd.DataFrame([
                    {"date": switch_date, "type":"BUY", "price": px_buy, "note":"No‑Regret → B&H"},
                    {"date": last_date,   "type":"SELL","price": px_sell, "note":"End"}
                ])

# --- Results UI ---
st.subheader("Wyniki")
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Benchmark (B&H) ROI", f"{info['bench_roi_pct']:.2f}%")
with c2: st.metric("Tryb", info["mode"])
with c3: st.metric("Próby (tested)", info["tested"])
with c4: st.metric("Bias", info.get("bias","none"))

st.write("**Najlepsze parametry**:", info["best_params"] if info["best_params"] is not None else "—")
if info["best_params"] is not None:
    st.metric("Model ROI (sum P&L na 100 USD/trade)", f"{info['best_model_roi_usd']:.2f} USD")

st.subheader("Dzisiejsza rekomendacja")
cA, cB, cC = st.columns(3)
with cA: st.metric("Akcja", today_rec.get("action","HOLD"))
with cB: st.metric("Cena", f"{today_rec.get('price', float(df['Zamkniecie'].iloc[-1])):.2f}")
with cC: st.write(today_rec.get("note",""))

# Price + signals
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df["Zamkniecie"], label="Close")
ax.plot(df.index, df["EMA100"], label="EMA100")
if short_mode == "15":
    ax.plot(df.index, df["EMA15"], label="EMA15")
elif short_mode == "40":
    ax.plot(df.index, df["EMA40"], label="EMA40")
else:
    ax.plot(df.index, df["EMA40"], label="EMAshort (dyn proxy=EMA40)")
bx, by, sx, sy, axx, axy = [], [], [], [], [], []
for _, r in sig_adj.iterrows():
    d = pd.to_datetime(r['date'])
    if d in df.index:
        px = float(df.loc[d,"Zamkniecie"])
        if r['type']=="BUY": bx.append(d); by.append(px)
        elif r['type']=="SELL": sx.append(d); sy.append(px)
        elif r['type']=="ADD": axx.append(d); axy.append(px)
if bx: ax.scatter(bx, by, marker='^', s=60, label="BUY")
if axx: ax.scatter(axx, axy, marker='o', s=40, label="ADD")
if sx: ax.scatter(sx, sy, marker='v', s=60, label="SELL")
if switch_date is not None:
    ax.axvline(pd.to_datetime(switch_date), linestyle='--')
ax.legend(); ax.set_title(f"Sygnały — short_ema={short_mode} — bias={info.get('bias','none')} — No‑Regret={'ON' if no_regret else 'OFF'}")
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

# Equity curves
fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(eq.index, eq["Model"], label="Model (No‑Regret włączony)" if no_regret else "Model")
ax2.plot(eq.index, eq["Buy&Hold"], label="Buy&Hold")
if switch_date is not None:
    ax2.axvline(pd.to_datetime(switch_date), linestyle='--')
ax2.set_title("Equity Curve – Model vs Buy&Hold (start=100)")
ax2.legend(); fig2.tight_layout()
st.pyplot(fig2, clear_figure=True)

# Downloads
st.subheader("Eksport")
st.download_button("Pobierz signals.csv", sig_adj.to_csv(index=False).encode("utf-8"), "signals.csv", "text/csv")
st.download_button("Pobierz trades.csv", trd.to_csv(index=False).encode("utf-8"), "trades.csv", "text/csv")



# =====================
# Interaktywny wykres (Plotly) + zakres dat
# =====================
import plotly.graph_objects as go
import pandas as pd

st.subheader("Zakres wykresu")
view_mode = st.radio("Wybór zakresu", ["Pełny", "Ostatnie N dni", "Zakres dat"], index=1, horizontal=True)
if view_mode == "Ostatnie N dni":
    n_days = st.slider("Ile ostatnich dni pokazać?", 10, 365, 90, step=5)
    df_view = df.copy().iloc[-n_days:]
elif view_mode == "Zakres dat":
    c1, c2 = st.columns(2)
    with c1:
        d_from = st.date_input("Od", value=pd.to_datetime(start).date())
    with c2:
        d_to = st.date_input("Do", value=df.index.max().date())
    df_view = df.loc[pd.to_datetime(d_from):pd.to_datetime(d_to)]
else:
    df_view = df.copy()

# Build Plotly figure
figp = go.Figure()
figp.add_trace(go.Scatter(x=df_view.index, y=df_view["Zamkniecie"], mode="lines", name="Close"))
figp.add_trace(go.Scatter(x=df_view.index, y=df_view.get("EMA100"), mode="lines", name="EMA100"))
if short_mode == "15":
    figp.add_trace(go.Scatter(x=df_view.index, y=df_view.get("EMA15"), mode="lines", name="EMA15"))
elif short_mode == "40":
    figp.add_trace(go.Scatter(x=df_view.index, y=df_view.get("EMA40"), mode="lines", name="EMA40"))
else:
    figp.add_trace(go.Scatter(x=df_view.index, y=df_view.get("EMA40"), mode="lines", name="EMAshort (dyn proxy=EMA40)"))

# Markers for signals (filtered to view)
if not sig.empty:
    sig_view = sig.copy()
    sig_view["date"] = pd.to_datetime(sig_view["date"])
    sig_view = sig_view[(sig_view["date"] >= df_view.index.min()) & (sig_view["date"] <= df_view.index.max())]
    buys = sig_view[sig_view["type"]=="BUY"]
    sells= sig_view[sig_view["type"]=="SELL"]
    adds = sig_view[sig_view["type"]=="ADD"]
    def px_on_date(d):
        try:
            return float(df.loc[d, "Zamkniecie"])
        except Exception:
            idx = df.index.get_indexer([d], method="nearest")[0]
            return float(df["Zamkniecie"].iloc[idx])
    if not buys.empty:
        figp.add_trace(go.Scatter(x=buys["date"], y=[px_on_date(d) for d in buys["date"]],
                                  mode="markers", name="BUY", marker_symbol="triangle-up", marker_size=10))
    if not adds.empty:
        figp.add_trace(go.Scatter(x=adds["date"], y=[px_on_date(d) for d in adds["date"]],
                                  mode="markers", name="ADD", marker_symbol="circle", marker_size=8))
    if not sells.empty:
        figp.add_trace(go.Scatter(x=sells["date"], y=[px_on_date(d) for d in sells["date"]],
                                  mode="markers", name="SELL", marker_symbol="triangle-down", marker_size=10))

figp.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"),
                   margin=dict(l=10,r=10,t=40,b=10),
                   height=500,
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

st.subheader("Wykres interaktywny (powiększanie/scroll)")
st.plotly_chart(figp, use_container_width=True)

# =====================
# Tabela sygnałów
# =====================
st.subheader("Tabela sygnałów (filtruje się zakresem wykresu)")
if sig.empty:
    st.info("Brak sygnałów do wyświetlenia.")
else:
    st.dataframe(sig_view.sort_values("date"), use_container_width=True, height=300)
    st.download_button("Pobierz tabelę sygnałów (CSV)",
                       sig_view.to_csv(index=False).encode("utf-8"),
                       "signals_filtered.csv", "text/csv")

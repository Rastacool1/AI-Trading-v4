import os, sys, traceback, time
import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Trader – Regime Adaptive RSI", layout="wide")
st.title("AI Trader – Regime Adaptive RSI (by SO)")
st.caption("Build: v3 (cache reset, safe switching, robust regime scan)")

def _mpl():
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt
    return plt

def _import_engine():
    try:
        from src.ai_trader_engine import (
            load_price_csv, run_engine_regime, compute_regime_thresholds,
            dynamic_thresholds, optimize_params_by_regime, grid_search_params
        )
        return load_price_csv, run_engine_regime, compute_regime_thresholds, dynamic_thresholds, optimize_params_by_regime, grid_search_params
    except Exception as e1:
        APP_DIR = os.path.dirname(os.path.abspath(__file__))
        SRC_DIR = os.path.join(APP_DIR, "src")
        if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
        try:
            from ai_trader_engine import (
                load_price_csv, run_engine_regime, compute_regime_thresholds,
                dynamic_thresholds, optimize_params_by_regime, grid_search_params
            )
            return load_price_csv, run_engine_regime, compute_regime_thresholds, dynamic_thresholds, optimize_params_by_regime, grid_search_params
        except Exception as e2:
            st.error("❌ Import silnika nieudany.")
            with st.expander("🔎 Diag", expanded=True):
                st.code(
                    f"APP_DIR={APP_DIR}\nSRC_DIR={SRC_DIR}\n"
                    f"sys.path[0:5]={sys.path[:5]}\n"
                    f"Pliki w ./src: {os.listdir(SRC_DIR) if os.path.isdir(SRC_DIR) else 'brak'}\n\n"
                    f"e1: {''.join(traceback.format_exception_only(type(e1), e1))}\n"
                    f"e2: {''.join(traceback.format_exception_only(type(e2), e2))}\n"
                )
            st.stop()

load_price_csv, run_engine_regime, compute_regime_thresholds, dynamic_thresholds, optimize_params_by_regime, grid_search_params = _import_engine()

STQ_PRESETS = {
    "S&P500 futures (continuous) — ES.c": "es.c",
    "Bitcoin — BTCUSD": "btcusd",
    "EUR/USD — EURUSD": "eurusd",
    "Złoto — XAUUSD": "xauusd",
    "Ropa WTI — CL.c": "cl.c",
    "WIG20 — WIG20": "wig20",
}

@st.cache_data(ttl=1800, show_spinner="Pobieram z Stooq...")
def load_from_stooq(symbol: str, interval: str = "d") -> pd.DataFrame:
    url = f"https://stooq.pl/q/d/l/?s={symbol.lower()}&i={interval}"
    return load_price_csv(url)

# Session store
if "data_store" not in st.session_state:
    st.session_state["data_store"] = {"df": None, "name": None, "ver": 0}

def set_dataset(df: pd.DataFrame, name: str):
    st.session_state["data_store"]["df"] = df
    st.session_state["data_store"]["name"] = name
    st.session_state["data_store"]["ver"] += 1
    if "regime_overrides" in st.session_state:
        del st.session_state["regime_overrides"]

def get_dataset():
    return st.session_state["data_store"]["df"], st.session_state["data_store"]["name"]

def reset_all():
    st.cache_data.clear()
    for k in list(st.session_state.keys()):
        if k not in ("data_store",):
            del st.session_state[k]
    st.session_state["data_store"] = {"df": None, "name": None, "ver": 0}
    st.experimental_rerun()

with st.sidebar:
    st.header("Parametry globalne")
    start = st.text_input("Start backtestu (YYYY-MM-DD)", "2025-01-01")
    entry_rsi = st.slider("RSI próg BUY", 10, 50, 30, 1)
    overheat_rsi = st.slider("RSI przegrzanie", 60, 90, 75, 1)
    exit_trigger_rsi = st.slider("RSI cross-down EXIT", 50, 90, 70, 1)
    min_hold_days = st.slider("Min dni trzymania", 1, 20, 7, 1)
    exit_relax_mult = st.slider("Mnożnik progu wyjścia", 0.5, 1.5, 0.8, 0.05)
    dd_stop = st.slider("Hard stop (DD od wejścia)", 0.05, 0.5, 0.20, 0.01)
    min_gap_days = st.slider("Refractory (dni między sygnałami)", 0, 20, 3, 1)

    st.subheader("ATR / Filtry")
    enable_regimes = st.checkbox("Włącz reżimy", True)
    use_atr_norm = st.checkbox("Normalizuj progi ATR", True)
    k_atr = st.slider("k * ATR (fallback)", 1.0, 4.0, 2.0, 0.1)
    use_filter_roc = st.checkbox("Filtr ROC (BUY wymaga ROC≥0)", True)
    use_filter_macd = st.checkbox("Filtr MACD (BUY: MACD≥signal, SELL: MACD≤signal)", True)
    use_filter_atr_band = st.checkbox("Filtr ATR-band (blokuj SELL przy wybiciach)", True)
    atr_band_mult = st.slider("ATR band mnożnik", 0.5, 2.0, 1.0, 0.1)

    st.subheader("Walidacje i koszty")
    buy_validate_mult = st.slider("BUY: mnożnik progu spadku", 0.5, 1.5, 1.0, 0.05)
    sell_validate_mult = st.slider("SELL: mnożnik progu spadku", 0.5, 1.5, 1.0, 0.05)
    alloc_per_trade = st.number_input("Kwota per transakcja (USD)", 10.0, 100000.0, 100.0, 10.0)
    fees_bps = st.slider("Prowizja (bps)", 0, 50, 4, 1)
    slippage_bps = st.slider("Poślizg (bps)", 0, 50, 4, 1)

    st.markdown("---")
    if st.button("🧹 Wyczyść cache i pamięć"):
        reset_all()

st.markdown("### Wgraj CSV (Stooq lub Date,Close,High,Low)")
uploaded = st.file_uploader("Plik instrumentu", type=["csv"], key=f"u_{st.session_state['data_store']['ver']}")

st.markdown("### Albo pobierz bezpośrednio ze Stooq")
col1, col2, col3 = st.columns([2,1,1])
with col1:
    preset = st.selectbox("Preset (opcjonalnie)", ["— wybierz —"] + list(STQ_PRESETS.keys()))
    custom_symbol = st.text_input("Własny symbol Stooq (np. es.c, btcusd, wig20)", "")
with col2:
    interval = st.selectbox("Interwał", ["d", "w", "m"], index=0)
with col3:
    fetch_btn = st.button("⬇️ Pobierz z Stooq")

if fetch_btn:
    try:
        symbol = ""
        if preset != "— wybierz —": symbol = STQ_PRESETS[preset]
        if custom_symbol.strip(): symbol = custom_symbol.strip()
        if not symbol:
            st.error("Wybierz preset albo podaj własny symbol Stooq.")
        else:
            df_stq = load_from_stooq(symbol, interval)
            set_dataset(df_stq, f"{symbol.upper()} ({interval})")
            st.success(f"Pobrano {len(df_stq)} wierszy dla symbolu: {symbol}")
    except Exception as e:
        st.exception(e)

if uploaded is not None:
    try:
        df_up = load_price_csv(uploaded)
        set_dataset(df_up, f"UPLOAD: {getattr(uploaded, 'name', 'plik.csv')}")
        st.success(f"Wczytano plik: {getattr(uploaded, 'name', 'plik.csv')} ({len(df_up)} wierszy)")
    except Exception as e:
        st.exception(e)

df, ds_name = get_dataset()
if df is not None:
    st.info(f"Źródło danych: **{ds_name}**")

    # Core thresholds
    try:
        core = dynamic_thresholds(df['ret'].dropna())
        with st.expander("Progi z pełnej historii (core)", expanded=False):
            st.write(core)
    except Exception as e:
        st.exception(e)

    # Regimes (stats)
    if enable_regimes:
        try:
            regs, stats = compute_regime_thresholds(df)
            st.markdown("#### Progi per reżim")
            st.dataframe(pd.DataFrame(stats).T)
        except Exception as e:
            st.exception(e)

    # --- Robust regime scan with progress ---
    st.markdown("---")
    st.subheader("Auto-optymalizacja parametrów per reżim")

    start_auto = st.text_input("Start do optymalizacji (YYYY-MM-DD)", value=str(df.index.min().date()))
    run_opt = st.button("🔎 Skanuj reżimy i zaproponuj parametry", key=f"opt_{st.session_state['data_store']['ver']}")
    out_opt = st.container()
    if run_opt:
        with out_opt:
            try:
                try:
                    _ = pd.to_datetime(start_auto)
                except Exception:
                    st.warning("Nieprawidłowa data startu – używam pierwszej daty z danych.")
                    start_auto = str(df.index.min().date())

                regimes_list = ['bear_lowvol','bear_midvol','bull_highvol','bull_midvol','bull_lowvol','side_highvol','side_midvol','side_lowvol']
                progress = st.progress(0.0, text="Skanowanie reżimów...")
                rows = []
                total = len(regimes_list)
                for i, r in enumerate(regimes_list, start=1):
                    progress.progress(i/total, text=f"Skanuję: {r} ({i}/{total})")
                    search = {"entry_rsi":[25,30,35], "exit_trigger_rsi":[65,70,75], "min_hold_days":[3,5,7], "exit_relax_mult":[0.7,0.8,0.9]}
                    # Prefer direct grid if available
                    if grid_search_params is not None:
                        best_p, best_m = grid_search_params(df=df, start_date=start_auto, regime_name=r, search=search)
                        if best_p is not None:
                            rows.append({"regime": r, **best_p, **best_m})
                    else:
                        rec = optimize_params_by_regime(df=df, start_date=start_auto, regimes=[r], search=search)
                        if rec is not None and not rec.empty:
                            rows.append(rec.iloc[0].to_dict())
                progress.empty()

                if len(rows)==0:
                    st.warning("Brak rekomendacji – możliwe, że w danych jest za mało obserwacji per reżim.")
                else:
                    import pandas as _pd
                    rec = _pd.DataFrame(rows)
                    st.success("Gotowe – rekomendacje poniżej.")
                    st.dataframe(rec)

                    overrides = {}
                    for _, row in rec.iterrows():
                        overrides[row["regime"]] = {
                            "entry_rsi": int(row["entry_rsi"]),
                            "exit_trigger_rsi": int(row["exit_trigger_rsi"]),
                            "min_hold_days": int(row["min_hold_days"]),
                            "exit_relax_mult": float(row["exit_relax_mult"]),
                        }
                    st.session_state["regime_overrides"] = overrides
                    st.info("Kliknij „Przelicz dla tego pliku”, aby użyć rekomendacji.")
            except Exception as e:
                st.exception(e)

    # --- Recalculate
    st.markdown("---")
    if st.button("Przelicz dla tego pliku"):
        try:
            overrides = st.session_state.get("regime_overrides", None)
            sig, trd, par, filt = run_engine_regime(
                df=df, start_date=start,
                entry_rsi=entry_rsi, overheat_rsi=overheat_rsi, exit_trigger_rsi=exit_trigger_rsi,
                min_hold_days=min_hold_days, exit_relax_mult=exit_relax_mult, dd_stop=dd_stop,
                buy_validate_mult=buy_validate_mult, sell_validate_mult=sell_validate_mult,
                alloc_per_trade=alloc_per_trade, enable_regimes=enable_regimes,
                use_atr_norm=use_atr_norm, k_atr=k_atr, fees_bps=fees_bps, slippage_bps=slippage_bps,
                min_gap_days=min_gap_days, use_filter_roc=use_filter_roc, use_filter_macd=use_filter_macd,
                use_filter_atr_band=use_filter_atr_band, atr_band_mult=atr_band_mult,
                regime_param_overrides=overrides
            )
            roi = 100.0 + (trd["pnl_usd"].sum() if not trd.empty else 0.0)
            st.subheader(f"ROI (alokacja 100 USD na trade): {roi:.2f} USD")

            try:
                plt = _mpl()
                fig, ax = plt.subplots(figsize=(12,6))
                df_view = df.loc[pd.to_datetime(start):]
                ax.plot(df_view.index, df_view['Zamkniecie'], label="Close")
                if not sig.empty:
                    idx = pd.to_datetime(sig['date'])
                    px = df['Zamkniecie'].reindex(idx)
                    buy_mask = sig['type'] == "BUY"
                    sell_mask = sig['type'] == "SELL"
                    if buy_mask.any():
                        ax.scatter(idx[buy_mask], px[buy_mask], marker='^', s=70, label="BUY")
                    if sell_mask.any():
                        ax.scatter(idx[sell_mask], px[sell_mask], marker='v', s=70, label="SELL")
                ax.legend(); ax.set_xlabel("Data"); ax.set_ylabel("Cena"); ax.set_title(f"Sygnały BUY/SELL – {ds_name}")
                st.pyplot(fig)
            except Exception as e:
                st.warning("Wykres: problem ze znacznikami – pokazuję same ceny.")
                st.line_chart(df_view['Zamkniecie'])

            st.subheader("Sygnały"); st.dataframe(sig)
            st.subheader("Transakcje"); st.dataframe(trd)
            st.subheader("Parametry per dzień"); st.dataframe(par)
            st.subheader("Odrzucone sygnały"); st.dataframe(filt)
        except Exception as e:
            st.exception(e)
else:
    st.info("Wgraj CSV albo pobierz dane ze Stooq powyżej, aby rozpocząć.")

with st.expander("🔧 Diagnostyka środowiska"):
    st.write("cwd:", os.getcwd())
    st.write("sys.path[0:5]:", sys.path[:5])
    st.write("Pliki w ./src:", os.listdir("src") if os.path.isdir("src") else "brak katalogu src")
    st.write("Session data:", st.session_state.get("data_store", {}))


import pandas as pd, numpy as np
from typing import Tuple, Dict

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = np.where(d>0, d, 0.0)
    dn = np.where(d<0, -d, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(n).mean()
    roll_dn = pd.Series(dn, index=series.index).rolling(n).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100/(1+rs))

def load_price_csv(csv_path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(csv_path_or_buffer)
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    dc = pick('data','date')
    cc = pick('zamkniecie','close','adj close','adj_close')
    hc = pick('najwyzszy','high')
    lc = pick('najnizszy','low')
    if dc is None or cc is None:
        raise ValueError("CSV must contain a date and close column (Data/Date, Zamkniecie/Close).")
    if hc is None or lc is None:
        base = pd.to_numeric(df[cc], errors='coerce')
        df['__high__'] = base * 1.001
        df['__low__']  = base * 0.999
        hc, lc = '__high__', '__low__'
    out = pd.DataFrame({
        'Data': pd.to_datetime(df[dc], errors='coerce'),
        'Zamkniecie': pd.to_numeric(df[cc], errors='coerce'),
        'Najwyzszy': pd.to_numeric(df[hc], errors='coerce'),
        'Najnizszy': pd.to_numeric(df[lc], errors='coerce'),
    }).dropna()
    out = out.sort_values('Data').set_index('Data')
    out['ret'] = out['Zamkniecie'].pct_change()
    out['EMA10']  = out['Zamkniecie'].ewm(span=10,  adjust=False).mean()
    out['EMA15']  = out['Zamkniecie'].ewm(span=15,  adjust=False).mean()
    out['EMA40']  = out['Zamkniecie'].ewm(span=40,  adjust=False).mean()
    out['EMA100'] = out['Zamkniecie'].ewm(span=100, adjust=False).mean()
    out['RSI14'] = rsi(out['Zamkniecie'], 14)
    return out

def _adaptive_halflife(vol, base_hl=90, min_hl=30, max_hl=240, alpha=1.0):
    vol_med = vol.rolling(365, min_periods=30).median()
    ratio = (vol_med / (vol.replace(0, np.nan))).clip(0.25, 4.0)
    hl = base_hl * (ratio ** alpha)
    return hl.clip(min_hl, max_hl)

def _ewm_by_hl(series, halflife):
    idx = series.index
    out = pd.Series(index=idx, dtype=float)
    prev = np.nan
    for t in range(len(idx)):
        x = series.iloc[t]
        hl = float(halflife.iloc[t]) if not np.isnan(halflife.iloc[t]) else 90.0
        alpha = 1 - np.exp(-np.log(2)/max(1.0, hl))
        prev = x if np.isnan(prev) else alpha*x + (1-alpha)*prev
        out.iloc[t] = prev
    return out

def adaptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    r = df['ret'].fillna(0.0)
    vol20 = r.rolling(20).std()
    hl = _adaptive_halflife(vol20)
    pos = r.where(r>0, 0.0)
    neg = (-r.where(r<0, 0.0))
    avg_up = _ewm_by_hl(pos, hl)
    avg_dn = _ewm_by_hl(neg, hl)
    sgn = np.sign(r.replace(0, np.nan))
    chg = (sgn != sgn.shift(1)).astype(float).fillna(0.0)
    p_chg = _ewm_by_hl(chg, hl).clip(1e-4, 0.9)
    L_est = 1.0 / p_chg
    L_up = L_est.clip(3, 20)
    L_dn = L_est.clip(3, 20)
    up_thr = (avg_up * L_up).rename("up_thr")
    dn_thr = (avg_dn * L_dn).rename("dn_thr")
    ema_short_span = (L_up/2.0).clip(6, 60).rename("ema_short_span")
    return pd.DataFrame({"up_thr": up_thr, "dn_thr": dn_thr, "L_up": L_up, "L_dn": L_dn, "hl": hl, "ema_short_span": ema_short_span})

def _ema_by_span(series: pd.Series, span_series: pd.Series, min_span=4, max_span=120):
    idx = series.index
    out = pd.Series(index=idx, dtype=float)
    prev = np.nan
    for t in range(len(idx)):
        x = series.iloc[t]
        n = float(span_series.iloc[t]) if not np.isnan(span_series.iloc[t]) else 20.0
        n = max(min_span, min(max_span, n))
        alpha = 2.0 / (n + 1.0)
        prev = x if np.isnan(prev) else alpha*x + (1-alpha)*prev
        out.iloc[t] = prev
    return out

def run_once(df_bt: pd.DataFrame, adapt_bt: pd.DataFrame,
             entry_rsi: int = 30, trend_entry_rsi_floor: int = 30,
             entry_up_mult: float = 0.5, exit_no_prog_mult: float = 0.3,
             min_hold_days: int = 4, dd_stop: float = 0.25, min_gap_days: int = 1,
             add_breakout_mult: float = 0.5, fail_safe_days: int = 45,
             alloc_per_trade: float = 100.0, max_adds_per_trade: int = 2,
             short_ema_mode: str = "40"):
    dates = df_bt.index
    prices = df_bt['Zamkniecie']
    rsi14  = df_bt['RSI14']
    ema10  = df_bt['EMA10']
    ema15  = df_bt['EMA15']
    ema40  = df_bt['EMA40']
    ema100 = df_bt['EMA100']

    if short_ema_mode == "15":
        ema_short = ema15
    elif short_ema_mode == "40":
        ema_short = ema40
    elif short_ema_mode == "dyn":
        ema_short = _ema_by_span(df_bt['Zamkniecie'], adapt_bt['ema_short_span'])
    else:
        raise ValueError("short_ema_mode must be one of: '15','40','dyn'")

    position=False; entry_price=None; entry_idx=None; peak_price=None
    adds=[]; last_signal_i=None; last_trade_day=None
    signals=[]; trades=[]

    def can_emit(i_idx): return (last_signal_i is None) or ((i_idx - last_signal_i) >= min_gap_days)

    for i in range(1, len(dates)):
        d = dates[i]; price = prices.iloc[i]
        up_thr=float(adapt_bt["up_thr"].iloc[i]); L_up=int(max(3,min(20,adapt_bt["L_up"].iloc[i]))); L_dn=int(max(3,min(20,adapt_bt["L_dn"].iloc[i])))

        # Entries
        if not position and can_emit(i):
            entered=False
            if (rsi14.iloc[i-1] >= entry_rsi) and (rsi14.iloc[i] < entry_rsi):
                position=True; entry_price=price; entry_idx=i; peak_price=price; adds=[]
                signals.append({"date": d, "type":"BUY", "price": float(price), "note": "Dip"})
                last_signal_i=i; last_trade_day=d; entered=True
            if not entered:
                look=max(2, L_dn); min_p=float(prices.iloc[i-look:i+1].min()); rise=(price/min_p - 1.0) if min_p>0 else 0.0
                rsi_ok=(rsi14.iloc[i] >= trend_entry_rsi_floor) and (rsi14.iloc[i] >= rsi14.iloc[i-2:i].mean())
                if (rise >= entry_up_mult * max(0.0, up_thr)) and rsi_ok:
                    position=True; entry_price=price; entry_idx=i; peak_price=price; adds=[]
                    signals.append({"date": d, "type":"BUY", "price": float(price), "note": "Trend"})
                    last_signal_i=i; last_trade_day=d; entered=True
            if not entered:
                days_no=9999 if last_trade_day is None else (d - last_trade_day).days
                if days_no >= fail_safe_days:
                    mom_ok=(prices.iloc[i] > prices.iloc[i-5:i].mean()) and (rsi14.iloc[i] >= rsi14.iloc[i-5:i].mean())
                    if mom_ok:
                        position=True; entry_price=price; entry_idx=i; peak_price=price; adds=[]
                        signals.append({"date": d, "type":"BUY", "price": float(price), "note": "FailSafe"})
                        last_signal_i=i; last_trade_day=d; entered=True
            if entered: continue

        # In-position
        if position:
            peak_price = max(peak_price, price) if peak_price is not None else price
            add_now=False
            if len(adds) < max_adds_per_trade:
                if ((i - entry_idx) > 0) and (((i - entry_idx) % max(3, L_up)) == 0):
                    trend_ok=prices.iloc[i] >= prices.iloc[i-max(3,L_up):i+1].mean(); rsi_ok=rsi14.iloc[i] >= rsi14.iloc[i-3:i].mean()
                    add_now = trend_ok and rsi_ok
                if not add_now:
                    ref=max([entry_price] + [a[1] for a in adds] + [peak_price]); progress=(price/ref - 1.0) if ref>0 else 0.0
                    if progress >= add_breakout_mult * max(0.0, up_thr): add_now=True
                if add_now and can_emit(i):
                    adds.append((i, price)); signals.append({"date": d, "type":"ADD", "price": float(price), "note": "Pyramid"}); last_signal_i=i

            base_idx=max(0, i - max(2, L_up)); base_p=float(prices.iloc[base_idx]); prog=(price/base_p - 1.0) if base_p>0 else 0.0
            no_prog = prog < (exit_no_prog_mult * max(0.0, up_thr))
            momentum_exit = price < ema10.iloc[i]
            candidate_exit = no_prog or momentum_exit

            long_trend_break = price < ema100.iloc[i]
            short_below_long = (ema_short.iloc[i] < ema100.iloc[i])
            candidate_exit = candidate_exit and long_trend_break and short_below_long

            dd_from_entry = 1.0 - (price/entry_price) if entry_price else 0.0
            valid = ((i - entry_idx) >= min(4, max(1, int(L_up/2)))) or (dd_from_entry >= dd_stop)

            if candidate_exit and valid and can_emit(i):
                pnl=(price/entry_price - 1.0)*100.0
                total_pnl=pnl + sum([(price/ap - 1.0)*100.0 for (_,ap) in adds])
                tag = f"Exit(EMA{('dyn' if short_ema_mode=='dyn' else short_ema_mode)}<EMA100)"
                signals.append({"date": d, "type":"SELL", "price": float(price), "note": tag})
                trades.append({"entry_date": dates[entry_idx], "exit_date": d,
                               "entry_price": float(entry_price), "exit_price": float(price),
                               "bars": int(i-entry_idx), "pnl_usd": float(total_pnl), "adds": len(adds)})
                position=False; entry_price=None; entry_idx=None; peak_price=None; adds=[]
                last_signal_i=i; last_trade_day=d

    return pd.DataFrame(signals), pd.DataFrame(trades)

def bench_roi(df: pd.DataFrame, start: str) -> float:
    view = df.loc[pd.to_datetime(start):]
    v0 = float(view['Zamkniecie'].iloc[0]); v1 = float(view['Zamkniecie'].iloc[-1])
    return (v1/v0 - 1.0) * 100.0

def random_optimize(df: pd.DataFrame, start="2023-01-01", n_samples=150, seed=11, short_ema_mode="40",
                    bias: str = "none"):
    rng = np.random.default_rng(seed)
    df_bt = df.loc[pd.to_datetime(start):].copy()
    adapt = adaptive_stats(df); adapt_bt = adapt.reindex(df_bt.index)
    bh = bench_roi(df, start)

    space = {
        "entry_rsi": [25, 30],
        "trend_entry_rsi_floor": [25],
        "entry_up_mult": [0.30, 0.35, 0.40, 0.45, 0.50],
        "exit_no_prog_mult": [0.15, 0.20, 0.25, 0.30],
        "min_gap_days": [0, 1],
        "fail_safe_days": [30],
        "add_breakout_mult": [0.5],
        "max_adds_per_trade": [2, 3, 4],
    }
    if bias == "momentum":
        space["entry_up_mult"] = [0.25, 0.30, 0.35, 0.40]
        space["exit_no_prog_mult"] = [0.15, 0.20]
        space["max_adds_per_trade"] = [3, 4]
        space["min_gap_days"] = [0]

    keys = list(space.keys())
    def sample_params():
        return {k: space[k][rng.integers(0, len(space[k]))] for k in keys}

    best=None; best_roi=-1e18; best_sig=pd.DataFrame(); best_trd=pd.DataFrame(); tested=0
    for _ in range(n_samples):
        tested += 1
        params = sample_params()
        sig, trd = run_once(df_bt, adapt_bt, **params, short_ema_mode=short_ema_mode)
        roi = float(trd['pnl_usd'].sum()) if not trd.empty else 0.0
        if roi > best_roi:
            best, best_roi, best_sig, best_trd = params, roi, sig, trd
        if roi >= bh:
            break

    if best_roi < bh:
        sig_bh = pd.DataFrame([
            {"date": df_bt.index[0], "type":"BUY", "price": float(df_bt['Zamkniecie'].iloc[0]), "note":"B&H fallback"},
            {"date": df_bt.index[-1], "type":"SELL","price": float(df_bt['Zamkniecie'].iloc[-1]), "note":"B&H fallback"}])
        pnl_bh = bh
        trd_bh = pd.DataFrame([{"entry_date": df_bt.index[0], "exit_date": df_bt.index[-1],
                                "entry_price": float(df_bt['Zamkniecie'].iloc[0]),
                                "exit_price": float(df_bt['Zamkniecie'].iloc[-1]),
                                "bars": len(df_bt)-1, "pnl_usd": float(pnl_bh), "adds": 0}])
        info = {"mode":"fallback_bh","bench_roi_pct": bh, "best_params": best, "best_model_roi_usd": best_roi, "tested": tested, "short_ema_mode": short_ema_mode, "bias": bias}
        return info, sig_bh, trd_bh

    info = {"mode":"model","bench_roi_pct": bh, "best_params": best, "best_model_roi_usd": best_roi, "tested": tested, "short_ema_mode": short_ema_mode, "bias": bias}
    return info, best_sig, best_trd

def simulate_today_recommendation(df: pd.DataFrame, start="2023-01-01",
                                  params: dict | None = None,
                                  short_ema_mode: str = "40") -> dict:
    if params is None:
        params = dict(entry_rsi=30, trend_entry_rsi_floor=30,
                      entry_up_mult=0.4, exit_no_prog_mult=0.2,
                      min_hold_days=4, dd_stop=0.25, min_gap_days=1,
                      add_breakout_mult=0.5, fail_safe_days=45,
                      alloc_per_trade=100.0, max_adds_per_trade=2)
    df_bt = df.loc[pd.to_datetime(start):].copy()
    if len(df_bt) < 50:
        return {"action":"HOLD","note":"Za mało danych do pewnej decyzji.","price": float(df_bt['Zamkniecie'].iloc[-1])}
    adapt = adaptive_stats(df)
    adapt_bt = adapt.reindex(df_bt.index)

    dates = df_bt.index
    prices = df_bt['Zamkniecie']
    rsi14  = df_bt['RSI14']
    ema10  = df_bt['EMA10']
    ema15  = df_bt['EMA15']
    ema40  = df_bt['EMA40']
    ema100 = df_bt['EMA100']

    if short_ema_mode == "15":
        ema_short = ema15
    elif short_ema_mode == "40":
        ema_short = ema40
    elif short_ema_mode == "dyn":
        ema_short = _ema_by_span(df_bt['Zamkniecie'], adapt_bt['ema_short_span'])
    else:
        ema_short = ema40

    position=False; entry_price=None; entry_idx=None; peak_price=None; adds=[]
    last_signal_i=None; last_trade_day=None

    def can_emit(i_idx, min_gap_days): return (last_signal_i is None) or ((i_idx - last_signal_i) >= min_gap_days)

    for i in range(1, len(dates)):
        d = dates[i]; price = prices.iloc[i]
        up_thr=float(adapt_bt["up_thr"].iloc[i])
        L_up=int(max(3, min(20, adapt_bt["L_up"].iloc[i])))
        L_dn=int(max(3, min(20, adapt_bt["L_dn"].iloc[i])))

        if not position and can_emit(i, params["min_gap_days"]):
            entered=False
            if (rsi14.iloc[i-1] >= params["entry_rsi"]) and (rsi14.iloc[i] < params["entry_rsi"]):
                position=True; entry_price=price; entry_idx=i; peak_price=price; adds=[]
                last_signal_i=i; last_trade_day=d; entered=True
            if not entered:
                look=max(2, L_dn); min_p=float(prices.iloc[i-look:i+1].min())
                rise=(price/min_p - 1.0) if min_p>0 else 0.0
                rsi_ok=(rsi14.iloc[i] >= params["trend_entry_rsi_floor"]) and (rsi14.iloc[i] >= rsi14.iloc[i-2:i].mean())
                if (rise >= params["entry_up_mult"] * max(0.0, up_thr)) and rsi_ok:
                    position=True; entry_price=price; entry_idx=i; peak_price=price; adds=[]
                    last_signal_i=i; last_trade_day=d; entered=True
            if not entered:
                days_no=9999 if last_trade_day is None else (d - last_trade_day).days
                if days_no >= params["fail_safe_days"]:
                    mom_ok=(prices.iloc[i] > prices.iloc[i-5:i].mean()) and (rsi14.iloc[i] >= rsi14.iloc[i-5:i].mean())
                    if mom_ok:
                        position=True; entry_price=price; entry_idx=i; peak_price=price; adds=[]
                        last_signal_i=i; last_trade_day=d; entered=True
            if entered: continue

        if position:
            peak_price = max(peak_price, price) if peak_price is not None else price
            add_now=False
            if len(adds) < params["max_adds_per_trade"]:
                if ((i - entry_idx) > 0) and (((i - entry_idx) % max(3, L_up)) == 0):
                    trend_ok=prices.iloc[i] >= prices.iloc[i-max(3,L_up):i+1].mean(); rsi_ok=rsi14.iloc[i] >= rsi14.iloc[i-3:i].mean()
                    add_now = trend_ok and rsi_ok
                if not add_now:
                    ref=max([entry_price] + [a[1] for a in adds] + [peak_price]); progress=(price/ref - 1.0) if ref>0 else 0.0
                    if progress >= params["add_breakout_mult"] * max(0.0, up_thr): add_now=True
                if add_now and can_emit(i, params["min_gap_days"]):
                    adds.append((i, price)); last_signal_i=i

    i = len(dates)-1
    d = dates[i]; price = float(prices.iloc[i])
    up_thr=float(adapt_bt["up_thr"].iloc[i])
    L_up=int(max(3, min(20, adapt_bt["L_up"].iloc[i])))
    L_dn=int(max(3, min(20, adapt_bt["L_dn"].iloc[i])))

    if not position:
        buy_dip = (rsi14.iloc[i-1] >= params["entry_rsi"]) and (rsi14.iloc[i] < params["entry_rsi"])
        look=max(2, L_dn); min_p=float(prices.iloc[i-look:i+1].min()); rise=(price/min_p - 1.0) if min_p>0 else 0.0
        rsi_ok=(rsi14.iloc[i] >= params["trend_entry_rsi_floor"]) and (rsi14.iloc[i] >= rsi14.iloc[i-2:i].mean())
        buy_trend = (rise >= params["entry_up_mult"] * max(0.0, up_thr)) and rsi_ok
        if buy_dip or buy_trend:
            return {"action":"BUY","note":"Wejście: sygnał dip/trend spełniony.","price": price}
        return {"action":"ACCUMULATE","note":"Brak wejścia; akumulacja/obserwacja.","price": price}
    else:
        base_idx=max(0, i - max(2, L_up)); base_p=float(prices.iloc[base_idx])
        prog=(price/base_p - 1.0) if base_p>0 else 0.0
        no_prog = prog < (params["exit_no_prog_mult"] * max(0.0, up_thr))
        momentum_exit = price < float(ema10.iloc[i])
        candidate_exit = no_prog or momentum_exit
        long_trend_break = price < float(ema100.iloc[i])
        short_below_long = (float(ema_short.iloc[i]) < float(ema100.iloc[i]))
        exit_ok = candidate_exit and long_trend_break and short_below_long

        can_add=False
        if len(adds) < params["max_adds_per_trade"]:
            if ((i - entry_idx) > 0) and (((i - entry_idx) % max(3, L_up)) == 0):
                trend_ok=prices.iloc[i] >= prices.iloc[i-max(3,L_up):i+1].mean()
                rsi_ok=rsi14.iloc[i] >= rsi14.iloc[i-3:i].mean()
                can_add = trend_ok and rsi_ok
            if not can_add:
                ref=max([entry_price] + [a[1] for a in adds] + [peak_price])
                progress=(price/ref - 1.0) if ref>0 else 0.0
                if progress >= params["add_breakout_mult"] * max(0.0, up_thr):
                    can_add=True

        if exit_ok:
            return {"action":"SELL","note":"Wyjście: dual‑EMA + brak progresu/momentum.","price": price}
        if can_add:
            return {"action":"ADD","note":"Dokładka: trend/momentum postępuje.","price": price}
        return {"action":"HOLD","note":"Trzymaj pozycję.","price": price}

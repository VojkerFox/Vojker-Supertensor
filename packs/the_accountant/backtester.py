# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import jax.numpy as jnp
import numpy as np  # Lisätty perus-NumPy nopeaan tiedonkäsittelyyn
from datetime import datetime
import sys
import os

# Poka-Yoke: Lisätään projektin juuri polkuun
sys.path.append(os.getcwd())

# IMPORTATAAN SE STANDARDI LOGIIKKA (Koskematon)
from packs.the_accountant.logic import analyze_signal_core
from packs.the_accountant.sniffer import calculate_wolff_lots

def run_pareto_backtest():
    if not mt5.initialize():
        print("KRIITTINEN VIRHE: MT5 yhteys epäonnistui!")
        return

    # PARETO-LISTA: 8 symbolia, Hopea mukana
    SYMBOLS = ["XAUUSD", "XAGUSD", "GBPUSD", "AUDUSD", "EURUSD", "USDCAD", "NZDUSD", "USDCHF"]
    RISK_USD = 250.0
    CANDLES = 1440 * 5 # 5 päivää

    print(f"\n{'='*85}\n THE ACCOUNTANT: PARETO AUDIT (NUMPY SPEED BRIDGE) \n{'='*85}")
    
    # 1. DATAN HAKU (Pre-allocation puhtaaseen NumPy-muotoon)
    np_data_m1 = {}
    np_data_m5 = {}
    sym_info = {}
    
    for s in SYMBOLS:
        m1 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M1, 0, CANDLES + 100)
        m5 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, CANDLES + 100)
        info = mt5.symbol_info(s)
        
        if m1 is None or m5 is None or info is None:
            print(f"VIRHE: Dataa ei saatu: {s}")
            return
            
        # Käytetään perus-Numpya loopin nopeuttamiseksi!
        np_data_m1[s] = np.array([[r['open'], r['high'], r['low'], r['close']] for r in m1])
        np_data_m5[s] = np.array([[r['open'], r['high'], r['low'], r['close']] for r in m5])
        
        sym_info[s] = {
            'tick_size': info.trade_tick_size,
            'tick_value': info.trade_tick_value,
            'point': info.point
        }

    stats = {s: {"net": 0.0, "trades": 0, "wins": 0} for s in SYMBOLS}
    active = {s: None for s in SYMBOLS}

    # 2. SIMULAATIO
    print("Analysoidaan Panama-prosessia (Pitäisi kestää vain sekunteja)...")
    
    for t in range(160, CANDLES):
        if t % 500 == 0:
            print(f"Progress: {t}/{CANDLES} kynttilää...", end='\r')

        tensor_list = []
        for s in SYMBOLS:
            # Slicing tapahtuu nyt salamannopealla NumPyllä
            m1_win = np_data_m1[s][t-30:t]
            m5_idx = t // 5
            m5_win = np_data_m5[s][m5_idx-30:m5_idx]
            
            # RNAI lasketaan perus-NumPyllä
            rnai_val = (m1_win[-1, 3] - m1_win[-10, 0]) / (np.std(m1_win[:, 3]) + 1e-6)
            rnai_layer = np.full((30, 4), rnai_val)
            tensor_list.append(np.stack([m1_win, m5_win, rnai_layer]))
            
        # TÄSSÄ TEHDÄÄN JAX-MUUNNOS VAIN KERRAN PER ASKEL!
        supertensor = jnp.array(np.stack(tensor_list))
        
        # AJETAAN STANDARDI LOGIIKKA (Tämä lentää, koska se on JIT-käännetty)
        signals, b_highs, b_lows = analyze_signal_core(supertensor)
        sig_list = signals.tolist()

        for i, s in enumerate(SYMBOLS):
            sig = sig_list[i]
            curr_bar = np_data_m1[s][t]
            c_high, c_low, c_close = float(curr_bar[1]), float(curr_bar[2]), float(curr_bar[3])
            si = sym_info[s]
            
            if active[s]:
                tr = active[s]
                hit_sl = (tr['type'] == "BUY" and c_low <= tr['sl']) or \
                         (tr['type'] == "SELL" and c_high >= tr['sl'])
                hit_tp = (tr['type'] == "BUY" and c_high >= tr['tp']) or \
                         (tr['type'] == "SELL" and c_low <= tr['tp'])
                
                if hit_sl or hit_tp:
                    exit_p = tr['sl'] if hit_sl else tr['tp']
                    diff = (exit_p - tr['entry']) if tr['type'] == "BUY" else (tr['entry'] - exit_p)
                    final_pnl = (diff / si['tick_size']) * si['tick_value'] * tr['lot']
                    
                    stats[s]['net'] += final_pnl
                    stats[s]['trades'] += 1
                    if final_pnl > 0: stats[s]['wins'] += 1
                    active[s] = None

            elif sig != 0:
                direction = "BUY" if sig == 1 else "SELL"
                sl = float(b_lows[i]) if direction == "BUY" else float(b_highs[i])
                risk_pts = abs(c_close - sl)
                
                min_risk = (20.0 if "XA" in s else 10.0) * si['point']
                if risk_pts < min_risk: continue
                
                lot = calculate_wolff_lots(s, RISK_USD)
                if lot <= 0: continue

                active[s] = {
                    "type": direction, "entry": c_close, "sl": sl, 
                    "tp": c_close + (risk_pts * 2) if direction == "BUY" else c_close - (risk_pts * 2),
                    "lot": lot
                }

    # 3. LOPPURAPORTTI
    print("\n" + "-"*55)
    print(f"{'SYMBOL':<10} | {'NET PNL':<12} | {'TRADES':<8} | {'WIN %':<8}")
    print("-" * 55)
    
    for s in sorted(SYMBOLS, key=lambda x: stats[x]['net'], reverse=True):
        d = stats[s]
        wr = (d['wins']/d['trades']*100) if d['trades'] > 0 else 0
        print(f"{s:<10} | {d['net']:>+10.2f}$ | {d['trades']:<8} | {wr:>5.1f}%")

if __name__ == "__main__":
    run_pareto_backtest()
    mt5.shutdown()
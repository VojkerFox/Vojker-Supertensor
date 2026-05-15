# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import numpy as np
import jax.numpy as jnp
from datetime import datetime
import sys
import os

sys.path.append(os.getcwd())

# TÄHÄN EI KOSKETA. Logiikka (M5BOS -> M1B&R) on täällä.
from packs.the_accountant.logic import analyze_signal_core
from packs.the_accountant.sniffer import calculate_wolff_lots

def run_full_strategy_with_audit():
    if not mt5.initialize():
        print("MT5 Yhteysvirhe!")
        return

    # 1. KOKO STRATEGIA: 8 SYMBOLIA
    SYMBOLS = ["XAUUSD", "XAGUSD", "GBPUSD", "AUDUSD", "EURUSD", "USDCAD", "NZDUSD", "USDCHF"]
    RISK_USD = 250.0
    CANDLES = 1440 * 5 # 5 päivää

    print(f"\n{'='*85}\n THE ACCOUNTANT: FULL STRATEGY + FORENSIC AUDIT \n{'='*85}")
    print("Haetaan data ja optimoidaan (Odota sekunti)...")
    
    np_data_m1 = {}
    np_data_m5 = {}
    sym_info = {}
    
    for s in SYMBOLS:
        m1 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M1, 0, CANDLES + 100)
        m5 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, CANDLES + 100)
        info = mt5.symbol_info(s)
        
        if m1 is None or m5 is None or info is None: return
            
        # M1:ssä Aika (0) talteen lokeja varten, OHLC (1:5) logiikalle
        np_data_m1[s] = np.array([[r['time'], r['open'], r['high'], r['low'], r['close']] for r in m1])
        # M5:ssä vain OHLC, jotta (8, 3, 30, 4) tensor pysyy ehjänä
        np_data_m5[s] = np.array([[r['open'], r['high'], r['low'], r['close']] for r in m5])
        
        sym_info[s] = {
            'tick_size': info.trade_tick_size,
            'tick_value': info.trade_tick_value,
            'point': info.point
        }

    stats = {s: {"net": 0.0, "trades": 0, "wins": 0} for s in SYMBOLS}
    active = {s: None for s in SYMBOLS}
    trade_log = {s: [] for s in SYMBOLS} # Musta laatikko auditointia varten

    print("Analysoidaan koko linjastoa...")
    
    for t in range(160, CANDLES):
        if t % 1000 == 0:
            print(f"Progress: {t}/{CANDLES} kynttilää...", end='\r')

        tensor_list = []
        for s in SYMBOLS:
            # Otetaan vain OHLC (indeksit 1-4) tensorille
            m1_ohlc = np_data_m1[s][t-30:t, 1:5]
            m5_idx = t // 5
            m5_ohlc = np_data_m5[s][m5_idx-30:m5_idx]
            
            # RNAI standardi
            rnai_val = (m1_ohlc[-1, 3] - m1_ohlc[-10, 0]) / (np.std(m1_ohlc[:, 3]) + 1e-6)
            rnai_layer = np.full((30, 4), rnai_val)
            tensor_list.append(np.stack([m1_ohlc, m5_ohlc, rnai_layer]))
            
        # (8, 3, 30, 4) SUPERTENSOR
        supertensor_np = np.stack(tensor_list)
        signals, b_highs, b_lows = analyze_signal_core(supertensor_np)
        sig_list = signals.tolist()

        for i, s in enumerate(SYMBOLS):
            sig = sig_list[i]
            curr_bar = np_data_m1[s][t]
            bar_time = datetime.fromtimestamp(curr_bar[0])
            c_high, c_low, c_close = float(curr_bar[2]), float(curr_bar[3]), float(curr_bar[4])
            si = sym_info[s]
            
            # --- EXIT VALVONTA ---
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
                    
                    trade_log[s].append({
                        'entry_time': tr['entry_time'],
                        'exit_time': bar_time,
                        'type': tr['type'],
                        'pnl': final_pnl,
                        'is_win': final_pnl > 0
                    })
                    active[s] = None

            # --- TÄYDELLINEN ENTRY LOGIIKKA (SINUN SÄÄNTÖSI) ---
            elif sig != 0:
                direction = "BUY" if sig == 1 else "SELL"
                sl = float(b_lows[i]) if direction == "BUY" else float(b_highs[i])
                risk_pts = abs(c_close - sl)
                
                # --- WOLFF PIP SCALING ---
                pip_size = si['point'] * 10.0
                risk_in_pips = risk_pts / pip_size
                is_metal = "XA" in s
                
                # Valuutoille sinun täydellinen 3.5 - 15 pip
                # Metalleille skaalattu 10 - 40 pip (jotta Salaman laatikko mahtuu hengittämään)
                min_pips = 10.0 if is_metal else 3.5
                max_pips = 40.0 if is_metal else 15.0
                
                if not (min_pips <= risk_in_pips <= max_pips): 
                    continue
                
                lot = calculate_wolff_lots(s, RISK_USD)
                if lot <= 0: continue

                active[s] = {
                    "type": direction, "entry": c_close, "sl": sl, 
                    "tp": c_close + (risk_pts * 2) if direction == "BUY" else c_close - (risk_pts * 2),
                    "lot": lot,
                    "entry_time": bar_time
                }

    # ---------------------------------------------------------
    # KOKONAISRAPORTTI
    # ---------------------------------------------------------
    print("\n\n" + "-"*55)
    print(f"{'SYMBOL':<10} | {'NET PNL':<12} | {'TRADES':<8} | {'WIN %':<8}")
    print("-" * 55)
    
    for s in sorted(SYMBOLS, key=lambda x: stats[x]['net'], reverse=True):
        d = stats[s]
        wr = (d['wins']/d['trades']*100) if d['trades'] > 0 else 0
        print(f"{s:<10} | {d['net']:>+10.2f}$ | {d['trades']:<8} | {wr:>5.1f}%")

    # ---------------------------------------------------------
    # XAUUSD FORENSIC AUDIT (MIKSI KULTA TEKEE TURSKAA?)
    # ---------------------------------------------------------
    print("\n" + "="*55)
    print(" XAUUSD AUTOPSY REPORT ")
    print("="*55)
    
    xau_losses = [t for t in trade_log["XAUUSD"] if not t['is_win']]
    xau_wins = [t for t in trade_log["XAUUSD"] if t['is_win']]
    
    if len(xau_losses) > 0:
        loss_hours = {}
        for l in xau_losses:
            h = l['entry_time'].hour
            loss_hours[h] = loss_hours.get(h, 0) + 1
        
        sorted_hours = sorted(loss_hours.items(), key=lambda x: x[1], reverse=True)
        print("Kellonajat, jolloin Kulta osuu useimmin Stop Lossiin (Saha-alueet):")
        for hour, count in sorted_hours[:5]:
            print(f"  Klo {hour:02d}:00 - {count} tappiota")

        avg_loss_dur = np.mean([(l['exit_time'] - l['entry_time']).total_seconds() / 60 for l in xau_losses])
        avg_win_dur = np.mean([(w['exit_time'] - w['entry_time']).total_seconds() / 60 for w in xau_wins]) if xau_wins else 0
        
        print(f"\nTappioiden keskimääräinen kesto: {avg_loss_dur:.1f} minuuttia.")
        print(f"Voittojen keskimääräinen kesto:  {avg_win_dur:.1f} minuuttia.")

if __name__ == "__main__":
    run_full_strategy_with_audit()
    mt5.shutdown()
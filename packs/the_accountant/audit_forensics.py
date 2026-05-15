# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import numpy as np
import jax.numpy as jnp
from datetime import datetime
import sys
import os

sys.path.append(os.getcwd())

from packs.the_accountant.logic import analyze_signal_core
from packs.the_accountant.sniffer import calculate_wolff_lots

def run_forensic_audit():
    if not mt5.initialize():
        print("MT5 Yhteysvirhe")
        return

    # Analysoidaan vain Kulta (Ongelman ydin)
    SYMBOLS = ["XAUUSD"] 
    RISK_USD = 250.0
    CANDLES = 1440 * 5 # 5 päivää

    print(f"\n{'='*85}\n WOLFF FORENSICS: XAUUSD AUTOPSY REPORT \n{'='*85}")
    
    np_data_m1 = {}
    np_data_m5 = {}
    sym_info = {}
    
    s = "XAUUSD"
    m1 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M1, 0, CANDLES + 100)
    m5 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, CANDLES + 100)
    info = mt5.symbol_info(s)
    
    # HUOM: Otetaan myös 'time' talteen indeksiin 0
    np_data_m1[s] = np.array([[r['time'], r['open'], r['high'], r['low'], r['close']] for r in m1])
    np_data_m5[s] = np.array([[r['open'], r['high'], r['low'], r['close']] for r in m5]) # M5 pysyy samana tensorille
    
    si = {
        'tick_size': info.trade_tick_size,
        'tick_value': info.trade_tick_value,
        'point': info.point
    }

    active = None
    trade_log = [] # Musta laatikko

    for t in range(160, CANDLES):
        # Erotetaan aika (0) ja OHLC (1:5) jotta logiikka ei mene rikki
        m1_ohlc = np_data_m1[s][t-30:t, 1:5]
        m5_idx = t // 5
        m5_ohlc = np_data_m5[s][m5_idx-30:m5_idx]
        
        rnai_val = (m1_ohlc[-1, 3] - m1_ohlc[-10, 0]) / (np.std(m1_ohlc[:, 3]) + 1e-6)
        rnai_layer = np.full((30, 4), rnai_val)
        
        # Rakennetaan (1, 3, 30, 4) tensor vain Kullalle
        supertensor_np = np.stack([np.stack([m1_ohlc, m5_ohlc, rnai_layer])])
        signals, b_highs, b_lows = analyze_signal_core(supertensor_np)
        
        sig = int(signals[0])
        curr_bar = np_data_m1[s][t]
        
        # Aika ja Hinnat
        bar_time = datetime.fromtimestamp(curr_bar[0])
        c_high, c_low, c_close = float(curr_bar[2]), float(curr_bar[3]), float(curr_bar[4])
        
        if active:
            hit_sl = (active['type'] == "BUY" and c_low <= active['sl']) or \
                     (active['type'] == "SELL" and c_high >= active['sl'])
            hit_tp = (active['type'] == "BUY" and c_high >= active['tp']) or \
                     (active['type'] == "SELL" and c_low <= active['tp'])
            
            if hit_sl or hit_tp:
                exit_p = active['sl'] if hit_sl else active['tp']
                diff = (exit_p - active['entry']) if active['type'] == "BUY" else (active['entry'] - exit_p)
                pnl = (diff / si['tick_size']) * si['tick_value'] * active['lot']
                
                # Tallennetaan kaupan tiedot
                trade_log.append({
                    'entry_time': active['entry_time'],
                    'exit_time': bar_time,
                    'type': active['type'],
                    'risk_pips': active['risk_pips'],
                    'pnl': pnl,
                    'is_win': pnl > 0
                })
                active = None

        elif sig != 0:
            direction = "BUY" if sig == 1 else "SELL"
            sl = float(b_lows[0]) if direction == "BUY" else float(b_highs[0])
            risk_pts = abs(c_close - sl)
            
            pip_size = si['point'] * 10.0
            risk_in_pips = risk_pts / pip_size
            
            if not (10.0 <= risk_in_pips <= 40.0): 
                continue
            
            lot = calculate_wolff_lots(s, RISK_USD)
            if lot <= 0: continue

            active = {
                "type": direction, "entry": c_close, "sl": sl, 
                "tp": c_close + (risk_pts * 2) if direction == "BUY" else c_close - (risk_pts * 2),
                "lot": lot,
                "entry_time": bar_time,
                "risk_pips": risk_in_pips
            }

    # --- RUUMIINAVAUS (RAPORTTI) ---
    losses = [t for t in trade_log if not t['is_win']]
    wins = [t for t in trade_log if t['is_win']]
    
    print("\n[ TARKASTUS 1: NOUDATETTIINKO SÄÄNTÖJÄ? ]")
    rule_breaks = [t for t in trade_log if not (10.0 <= t['risk_pips'] <= 40.0)]
    print(f"Sääntörikkomuksia (Risk Pips < 10 tai > 40): {len(rule_breaks)} kpl")
    if len(rule_breaks) == 0:
        print("-> THE ACCOUNTANT noudatti prosessia 100% matemaattisella tarkkuudella.")

    print(f"\n[ TARKASTUS 2: TAPPIOIDEN AIKA-ANALYYSI ({len(losses)} tappiota) ]")
    loss_hours = {}
    for l in losses:
        h = l['entry_time'].hour
        loss_hours[h] = loss_hours.get(h, 0) + 1
    
    # Sortataan tunnit pahimmasta parhaaseen
    sorted_hours = sorted(loss_hours.items(), key=lambda x: x[1], reverse=True)
    print("Eniten tappioita generoivat tunnit (Saha-alueet):")
    for hour, count in sorted_hours[:5]:
        print(f"Klo {hour:02d}:00 - {count} tappiota")

    print(f"\n[ TARKASTUS 3: KAUPAN KESTO (Whipsaw Test) ]")
    avg_loss_duration = np.mean([(l['exit_time'] - l['entry_time']).total_seconds() / 60 for l in losses]) if losses else 0
    avg_win_duration = np.mean([(w['exit_time'] - w['entry_time']).total_seconds() / 60 for w in wins]) if wins else 0
    
    print(f"Tappiot osuivat Stop Lossiin keskimäärin: {avg_loss_duration:.1f} minuutissa.")
    print(f"Voitot osuivat Take Profitiin keskimäärin: {avg_win_duration:.1f} minuutissa.")
    
    net_pnl = sum([t['pnl'] for t in trade_log])
    print(f"\nXAUUSD LOPPUTULOS: {net_pnl:+.2f}$ (Trades: {len(trade_log)})")

if __name__ == "__main__":
    run_forensic_audit()
    mt5.shutdown()
# -*- coding: utf-8 -*-
"""
VOJKER BACKTEST ENGINE (MT5) - Advanced Institutional Reporter with Pareto
Simuloi live-ympäristö kohti määritettyä loppuaikaa askel askeleelta tarkan PnL-seurannan kanssa.
Laskee Profit Factorin, Drawdownin, kauppojen keston ja Pareto-juurisyyt tiimiesitystä varten.
"""
import sys
import os
import time
import argparse
import json
from datetime import datetime
import MetaTrader5 as mt5
import numpy as np
import jax.numpy as jnp

# Pakotetaan projektin juuri hakupolkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from packs.wolfpack_alpha.logic import analyze_signal_core
from scripts.panama_fsm import PanamaFSM

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "EURJPY"]

def get_historical_tensor(shift_pos, bars=30):
    """
    Toimii kuten MT5Adapter, mutta hakee datan 'shift_pos' minuuttia menneisyydestä (suhteessa nykyhetkeen).
    """
    raw_symbol_containers = []
    symbol_aggressions = []
    
    for s in SYMBOLS:
        r_m1 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M1, shift_pos, bars)
        
        m5_shift = int(shift_pos / 5)
        r_m5 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, m5_shift, bars)
        
        if r_m1 is None or r_m5 is None or len(r_m1) < bars or len(r_m5) < bars:
            return None
            
        m1_data = np.array([[r[1], r[2], r[3], r[4]] for r in r_m1], dtype=np.float32)
        m5_data = np.array([[r[1], r[2], r[3], r[4]] for r in r_m5], dtype=np.float32)
        
        last_candle = r_m1[-1]
        net_aggression = (last_candle[4] - last_candle[1]) * last_candle[5] 
        symbol_aggressions.append(net_aggression)
        
        raw_symbol_containers.append((m1_data, m5_data))

    market_avg_aggression = np.mean(symbol_aggressions)
    
    all_symbol_data = []
    for i, (m1_data, m5_data) in enumerate(raw_symbol_containers):
        rnai_val = symbol_aggressions[i] - market_avg_aggression
        q_data = np.ones_like(m1_data, dtype=np.float32) * rnai_val
        symbol_stack = np.stack([m1_data, m5_data, q_data], axis=0)
        all_symbol_data.append(symbol_stack)
        
    return jnp.array(all_symbol_data) if len(all_symbol_data) == 8 else None

def calculate_shift_parameters(start_str, end_str):
    """
    Muuntaa Y-m-dTH:i kalenteriajat (esim. 2026-05-01T08:00) 
    niiksi MT5:n 'shift_pos' minuuteiksi, joita alkuperäinen funktio vaatii.
    """
    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%dT%H:%M")
        end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H:%M")
        now_dt = datetime.now()

        # Kuinka monta minuuttia aloitus- ja lopetushetket ovat nykyhetkestä taaksepäin
        start_shift = int((now_dt - start_dt).total_seconds() / 60)
        end_shift = int((now_dt - end_dt).total_seconds() / 60)

        # Varmistetaan, että aloitus on kauempana menneisyydessä kuin lopetus
        if start_shift < end_shift:
            start_shift, end_shift = end_shift, start_shift
            
        # Varmistetaan ettei yritetä hakea tulevaisuutta
        start_shift = max(0, start_shift)
        end_shift = max(0, end_shift)

        history_minutes = start_shift - end_shift
        return history_minutes, start_shift, end_shift
    except Exception as e:
        print(f"[VIRHE] Aikaleimojen laskenta epäonnistui: {e}")
        return None, None, None

def run_backtest(start_time=None, end_time=None, symbol_filter=None, fallback_history=181440):
    
    # Ratkaistaan aikaparametrit
    if start_time and end_time:
        history_minutes, start_shift, end_shift = calculate_shift_parameters(start_time, end_time)
        if history_minutes is None:
            return
        print(f"> Aikaikkuna asetettu: {start_time} -> {end_time} ({history_minutes} min)")
    else:
        history_minutes = fallback_history
        start_shift = history_minutes
        end_shift = 0
        print(f"> Simuloidaan kiinteä historia: {history_minutes} min")

    if not mt5.initialize():
        print("> [VIRHE] MT5 Initialization failed.")
        return

    fsm = PanamaFSM()
    
    trades_executed = 0
    trade_pips_history = []
    durations = []
    entry_times = {} 
    trade_records = []
    
    wins = 0
    losses = 0
    total_pips = 0.0
    start_cpu_time = time.time()

    # Luuppi alkaa menneisyydestä (start_shift) ja kulkee kohti loppuaikaa (end_shift)
    for step in range(start_shift, end_shift - 1, -1):
        tensor = get_historical_tensor(step)
        
        if tensor is None:
            continue
            
        signal_mask, box_highs, box_lows = analyze_signal_core(tensor)
        
        current_prices = tensor[:, 0, -1, 3]
        rnai_values = tensor[:, 2, -1, 0]

        actions = fsm.update(signal_mask, box_highs, box_lows, current_prices, rnai_values)

        for i, act in enumerate(actions):
            if act:
                symbol = SYMBOLS[i]
                
                # Jos käyttöliittymästä valittiin tietty valuutta, ohitetaan muut
                if symbol_filter and symbol_filter != "ALL" and symbol != symbol_filter:
                    continue
                
                if "EXECUTE" in act or "LOCKED" in act or "EXIT" in act or "REACHED" in act:
                    timestamp = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, step, 1)[0][0]
                    human_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp))
                    
                    if "EXECUTE" in act:
                        trades_executed += 1
                        entry_times[i] = step 
                        print(f"[{human_time}] {symbol} -> EXECUTE (Long/Short entered)")
                    elif "LOCKED" in act:
                        print(f"[{human_time}] {symbol} -> ARMED (Signal locked)")
                    elif "REACHED" in act:
                        print(f"[{human_time}] {symbol} -> MANAGE (Target reached)")
                    elif "EXIT" in act:
                        if i in entry_times:
                            duration = entry_times[i] - step
                            durations.append(duration)
                            del entry_times[i]
                        else:
                            duration = 0
                        
                        direction = int(fsm.directions[i])
                        entry_price = float(fsm.lock_prices[i])
                        exit_price = float(current_prices[i])
                        
                        multiplier = 100 if "JPY" in symbol else 10000
                        
                        if direction == 1:
                            pips = (exit_price - entry_price) * multiplier
                        elif direction == -1:
                            pips = (entry_price - exit_price) * multiplier
                        else:
                            pips = 0.0
                            
                        total_pips += pips
                        trade_pips_history.append(pips)
                        
                        trade_records.append({
                            "symbol": symbol,
                            "pips": pips,
                            "duration": duration,
                            "direction": "LONG" if direction == 1 else "SHORT"
                        })
                        
                        if pips > 0:
                            wins += 1
                            print(f"[{human_time}] {symbol} -> EXIT (Voitto: +{pips:.1f} pips)")
                        else:
                            losses += 1
                            print(f"[{human_time}] {symbol} -> EXIT (Tappio: {pips:.1f} pips)")

    exec_time = time.time() - start_cpu_time

    # --- ADVANCED METRICS CALCULATIONS ---
    wins_list = [p for p in trade_pips_history if p > 0]
    losses_list = [p for p in trade_pips_history if p <= 0]
    
    gross_profit = sum(wins_list)
    gross_loss = abs(sum(losses_list))
    
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
    win_rate = (wins / trades_executed * 100) if trades_executed > 0 else 0.0
    
    avg_win = np.mean(wins_list) if len(wins_list) > 0 else 0.0
    avg_loss = np.mean(losses_list) if len(losses_list) > 0 else 0.0
    
    equity_curve = np.cumsum([0] + trade_pips_history)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    avg_duration = np.mean(durations) if len(durations) > 0 else 0.0
    
    # --- PARETO ANALYTICS CALCULATIONS ---
    sorted_losses = sorted(losses_list)
    
    num_p20 = max(1, int(len(sorted_losses) * 0.20))
    p20_losses = sorted_losses[:num_p20]
    p20_sum = abs(sum(p20_losses))
    p20_pct_of_total_loss = (p20_sum / gross_loss * 100) if gross_loss > 0 else 0.0
    
    num_p4 = max(1, int(len(sorted_losses) * 0.04))
    p4_losses = sorted_losses[:num_p4]
    p4_sum = abs(sum(p4_losses))
    p4_pct_of_total_loss = (p4_sum / gross_loss * 100) if gross_loss > 0 else 0.0

    symbol_losses = {}
    for t in trade_records:
        if t["pips"] <= 0:
            symbol_losses[t["symbol"]] = symbol_losses.get(t["symbol"], 0.0) + abs(t["pips"])
            
    worst_symbol = max(symbol_losses, key=symbol_losses.get) if symbol_losses else "N/A"
    worst_symbol_loss = symbol_losses.get(worst_symbol, 0.0)

    # --- JSON SARJALLISTAMINEN (Korvaa vanhan ASCII-raportin) ---
    
    report_data = {
        "trades_executed": int(trades_executed),
        "win_rate": round(float(win_rate), 1),
        "wins": int(wins),
        "losses": int(losses),
        "avg_duration": round(float(avg_duration), 1),
        "profit_factor": round(float(profit_factor), 2),
        "avg_win": round(float(avg_win), 1),
        "avg_loss": round(float(avg_loss), 1),
        "max_dd": round(float(max_dd), 1),
        "p20_trades": int(num_p20),
        "p20_pct": round(float(p20_pct_of_total_loss), 1),
        "p4_trades": int(num_p4),
        "p4_pct": round(float(p4_pct_of_total_loss), 1),
        "worst_symbol": str(worst_symbol),
        "worst_loss": round(float(worst_symbol_loss), 1),
        "exec_time": round(float(exec_time), 2),
        "total_pips": round(float(total_pips), 1)
    }

    # Tulostetaan JSON-rajapinta selaimeen
    print("\n===VOJKER_JSON===")
    print(json.dumps(report_data))
    
    mt5.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, help='ISO-alkuaika (Y-m-dTH:i)')
    parser.add_argument('--end', type=str, help='ISO-loppuaika (Y-m-dTH:i)')
    parser.add_argument('--symbol', type=str, default='ALL', help='Valuuttapari')
    
    args = parser.parse_args()

    run_backtest(start_time=args.start, end_time=args.end, symbol_filter=args.symbol)
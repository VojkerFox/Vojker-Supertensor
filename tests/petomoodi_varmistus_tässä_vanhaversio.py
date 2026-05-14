# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp

@jax.jit
def analyze_signal_core(tensor):
    """
    Panama Process Phase 2: Triple Quantile Gate (TQG)
    Input: (8, 3, 30, 4) Supertensor
    Output: signals (8,) - 0: Ei signaalia, 1: LONG, 2: SHORT
            box_highs (8,) - Salaman yläraja (Breakout-taso)
            box_lows (8,) - Salaman alaraja (Invalidaatiotaso)
    """
    # vmap monistaa logiikan kaikille 8 symbolille yhtä aikaa
    vmapped_logic = jax.vmap(process_symbol_logic)
    return vmapped_logic(tensor)

def process_symbol_logic(symbol_tensor):
    """
    Yksittäisen symbolin (3, 30, 4) analyysi.
    K=0: M1, K=1: M5, K=2: RNAI Context
    """
    m1 = symbol_tensor[0]  # Nopeus
    m5 = symbol_tensor[1]  # Rakenne
    rnai = symbol_tensor[2, -1, 0] # Viimeisin RNAI-arvo

    # 1. GATE I: Volatility Filter (Pareto 20%)
    m1_ranges = m1[:, 1] - m1[:, 2] # High - Low
    avg_volatility = jnp.mean(m1_ranges)
    current_vol = m1_ranges[-1]
    gate1 = current_vol > (avg_volatility * 1.5) # Vain poikkeuksellinen liike

    # 2. GATE II: Structural Alignment (Pareto 4%)
    m1_up = m1[-1, 3] > m1[-1, 0] # Close > Open
    m5_up = m5[-1, 3] > m5[-1, 0]
    
    m1_down = m1[-1, 3] < m1[-1, 0] # Close < Open
    m5_down = m5[-1, 3] < m5[-1, 0]
    
    gate2_long = jnp.logical_and(m1_up, m5_up)
    gate2_short = jnp.logical_and(m1_down, m5_down)

    # 3. GATE III: RNAI Aggression Filter (Signal Core 0.8%)
    gate3_long = rnai > 1.0 # Positiivinen RNAI = Ostoaggressio
    gate3_short = rnai < -1.0 # Negatiivinen RNAI = Myyntiaggressio

    # Lopullinen päätös: Kaikkien porttien on oltava auki
    is_long = jnp.logical_and(jnp.logical_and(gate1, gate2_long), gate3_long)
    is_short = jnp.logical_and(jnp.logical_and(gate1, gate2_short), gate3_short)
    
    # 4. SALAMAN LAATIKKO (3 viimeisimmän kynttilän Swing High & Swing Low)
    box_high = jnp.max(m1[-3:, 1]) # Keltaisen laatikon katto
    box_low = jnp.min(m1[-3:, 2])  # Keltaisen laatikon lattia

    signal = jnp.where(is_long, 1, jnp.where(is_short, 2, 0))
    
    return signal, box_high, box_low



##UUSI VERSIO

# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp

@jax.jit
def analyze_signal_core(tensor):
    """
    Panama Process Phase 2: Triple Quantile Gate (TQG)
    Input: (8, 3, 30, 4) Supertensor
    Output: signals (8,) - 0: Ei signaalia, 1: LONG, 2: SHORT
            box_highs (8,) - Salaman yläraja (Breakout-taso)
            box_lows (8,) - Salaman alaraja (Invalidaatiotaso)
    """
    # vmap monistaa logiikan kaikille 8 symbolille yhtä aikaa
    vmapped_logic = jax.vmap(process_symbol_logic)
    return vmapped_logic(tensor)

def process_symbol_logic(symbol_tensor):
    """
    Yksittäisen symbolin (3, 30, 4) analyysi.
    K=0: M1, K=1: M5, K=2: RNAI Context
    """
    m1 = symbol_tensor[0]  # Nopeus
    m5 = symbol_tensor[1]  # Rakenne
    rnai = symbol_tensor[2, -1, 0] # Viimeisin RNAI-arvo

    # 1. GATE I: Volatility Filter (Pareto 20%)
    m1_ranges = m1[:, 1] - m1[:, 2] # High - Low
    avg_volatility = jnp.mean(m1_ranges)
    current_vol = m1_ranges[-1]
    gate1 = current_vol > (avg_volatility * 1.5) # Vain poikkeuksellinen liike

    # 2. GATE II: Structural Alignment (Pareto 4%)
    m1_up = m1[-1, 3] > m1[-1, 0] # Close > Open
    m5_up = m5[-1, 3] > m5[-1, 0]
    
    m1_down = m1[-1, 3] < m1[-1, 0] # Close < Open
    m5_down = m5[-1, 3] < m5[-1, 0]
    
    gate2_long = jnp.logical_and(m1_up, m5_up)
    gate2_short = jnp.logical_and(m1_down, m5_down)

    # 3. GATE III: RNAI Aggression Filter (Signal Core 0.8%)
    gate3_long = rnai > 1.0 # Positiivinen RNAI = Ostoaggressio
    gate3_short = rnai < -1.0 # Negatiivinen RNAI = Myyntiaggressio

    # Lopullinen päätös: Kaikkien porttien on oltava auki
    is_long = jnp.logical_and(jnp.logical_and(gate1, gate2_long), gate3_long)
    is_short = jnp.logical_and(jnp.logical_and(gate1, gate2_short), gate3_short)
    
    # 4. SALAMAN LAATIKKO (3 viimeisimmän kynttilän Swing High & Swing Low)
    box_high = jnp.max(m1[-3:, 1]) # Keltaisen laatikon katto
    box_low = jnp.min(m1[-3:, 2])  # Keltaisen laatikon lattia

    # JAX int32 -korjaus (Ainoa minun lisäämäni asia, jotta testit toimivat)
    val_long = jnp.array(1, dtype=jnp.int32)
    val_short = jnp.array(2, dtype=jnp.int32)
    val_idle = jnp.array(0, dtype=jnp.int32)
    signal = jnp.where(is_long, val_long, jnp.where(is_short, val_short, val_idle))
    
    return signal, box_high, box_low



## backtestMT5 vanha koodi

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import json
from datetime import datetime
import MetaTrader5 as mt5
import numpy as np
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from packs.wolfpack_alpha.logic import analyze_signal_core
from scripts.panama_fsm import PanamaFSM

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "EURJPY"]

def get_historical_tensor(shift_pos, bars=30):
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

    market_avg = np.mean(symbol_aggressions)
    all_symbol_data = []
    for i, (m1_data, m5_data) in enumerate(raw_symbol_containers):
        rnai_val = symbol_aggressions[i] - market_avg
        q_data = np.ones_like(m1_data, dtype=np.float32) * rnai_val
        all_symbol_data.append(np.stack([m1_data, m5_data, q_data], axis=0))
    return jnp.array(all_symbol_data)

def run_backtest(fallback_history=7199):
    if not mt5.initialize(): return
    fsm = PanamaFSM()
    
    # --- AUDIT METRIIKAT ---
    stats = {"trades": 0, "wins": 0, "losses": 0, "total_pips": 0.0, "gross_profit": 0.0, "gross_loss": 0.0}
    trade_log = []
    
    print(f"=== VOJKER BACKTEST AUDIT MODE (Cpk 3.0) ===")
    print(f"LOG: Aloitetaan haku: {fallback_history} minuuttia menneisyyteen...")
    start_time_bench = time.time()

    for step in range(fallback_history, -1, -1):
        tensor = get_historical_tensor(step)
        if tensor is None: continue
            
        signal_mask, box_highs, box_lows = analyze_signal_core(tensor)
        current_prices = tensor[:, 0, -1, 3]
        rnai_values = tensor[:, 2, -1, 0]
        
        # Aikaleima
        candle_time = mt5.copy_rates_from_pos(SYMBOLS[0], mt5.TIMEFRAME_M1, step, 1)[0][0]
        human_time = datetime.fromtimestamp(candle_time).strftime('%d.%m %H:%M')
        
        # HUD-skrollaus
        sys.stdout.write(f"\r[AUDIT] {human_time} | Pips: {stats['total_pips'] :+.1f} | Win%: {(stats['wins']/max(1,stats['trades'])*100):.1f}% | Active: {sum(1 for d in fsm.directions if d != 0)}")
        sys.stdout.flush()

        actions = fsm.update(signal_mask, box_highs, box_lows, current_prices, rnai_values)

        for i, act in enumerate(actions):
            if act:
                symbol = SYMBOLS[i]
                if "EXECUTE" in act:
                    print(f"\n  [ENTRY] {human_time} | {symbol:7} | Price: {current_prices[i]:.5f}")
                elif "EXIT" in act:
                    # Lasketaan tulos
                    direction = fsm.directions[i]
                    entry_p = fsm.lock_prices[i]
                    exit_p = current_prices[i]
                    mult = 100 if "JPY" in symbol else 10000
                    pips = (exit_p - entry_p) * mult * direction
                    
                    stats["trades"] += 1
                    stats["total_pips"] += pips
                    if pips > 0:
                        stats["wins"] += 1
                        stats["gross_profit"] += pips
                        color = "\033[92m" # Green
                    else:
                        stats["losses"] += 1
                        stats["gross_loss"] += abs(pips)
                        color = "\033[91m" # Red
                    
                    print(f"  {color}[EXIT ] {human_time} | {symbol:7} | Result: {pips:+.1f} pips (Total: {stats['total_pips']:.1f})\033[0m")

    # --- LOPPURAPORTTI ---
    pf = stats["gross_profit"] / max(0.1, stats["gross_loss"])
    win_rate = (stats["wins"] / max(1, stats["trades"])) * 100
    duration = time.time() - start_time_bench

    print("\n\n" + "="*50)
    print(f" FINAL AUDIT REPORT - VOJKER PHASE 2.1")
    print("="*50)
    print(f" Total Trades:    {stats['trades']}")
    print(f" Win Rate:        {win_rate:.1f}%")
    print(f" Total Pips:      {stats['total_pips']:.1f}")
    print(f" Profit Factor:   {pf:.2f}")
    print(f" Execution Time:  {duration:.2f}s")
    print("="*50)
    
    mt5.shutdown()

if __name__ == "__main__":
    run_backtest(2880)

# -*- coding: utf-8 -*-
import sys
import os
import time
from datetime import datetime
import MetaTrader5 as mt5
import numpy as np
import jax.numpy as jnp
import pandas as pd

# Lisätään polut moduuleille
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from packs.wolfpack_alpha.logic import analyze_signal_core
from scripts.panama_fsm import PanamaFSM
from adapters.pg_adapter import PostgresAdapter

def get_all_marketwatch_symbols():
    symbols = mt5.symbols_get()
    return [s.name for s in symbols if s.visible]

def get_historical_tensor(symbols, shift_pos, bars=30):
    raw_symbol_containers = []
    symbol_aggressions = []
    actual_count = len(symbols)
    
    for s in symbols:
        r_m1 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M1, shift_pos, bars)
        m5_shift = int(shift_pos // 5)
        r_m5 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, m5_shift, bars)
        
        if r_m1 is None or r_m5 is None or len(r_m1) < bars or len(r_m5) < bars:
            return None
            
        m1_data = np.array([[r[1], r[2], r[3], r[4]] for r in r_m1], dtype=np.float32)
        m5_data = np.array([[r[1], r[2], r[3], r[4]] for r in r_m5], dtype=np.float32)
        
        last_candle = r_m1[-1]
        # Alkuperäinen aito RNAI-aggressio laskenta
        net_aggression = (last_candle[4] - last_candle[1]) * last_candle[5] 
        symbol_aggressions.append(net_aggression)
        raw_symbol_containers.append((m1_data, m5_data))

    # Padding JAX-moottoria varten (aina 8 paikkaa)
    while len(raw_symbol_containers) < 8:
        raw_symbol_containers.append((np.zeros((bars, 4)), np.zeros((bars, 4))))
        symbol_aggressions.append(0.0)

    market_avg = np.mean(symbol_aggressions[:actual_count]) if actual_count > 0 else 0
    all_symbol_data = []
    for i in range(8):
        m1_data, m5_data = raw_symbol_containers[i]
        rnai_val = symbol_aggressions[i] - market_avg if i < actual_count else 0.0
        q_data = np.ones_like(m1_data, dtype=np.float32) * rnai_val
        all_symbol_data.append(np.stack([m1_data, m5_data, q_data], axis=0))
        
    return jnp.array(all_symbol_data)

def run_backtest(fallback_history=2880):
    if not mt5.initialize(): 
        print("MT5 Connection Error")
        return
    
    db = PostgresAdapter()
    all_symbols = get_all_marketwatch_symbols()
    batches = [all_symbols[i:i + 8] for i in range(0, len(all_symbols), 8)]
    
    trade_log = []
    stats = {"trades": 0, "wins": 0, "losses": 0, "total_pips": 0.0, "gross_profit": 0.0, "gross_loss": 0.0}
    start_time_bench = time.time()

    print(f"=== VOJKER DATABASE & PARETO AUDIT (Symbols: {len(all_symbols)}) ===")
    
    for b_idx, current_symbols in enumerate(batches):
        fsm = PanamaFSM()
        for step in range(fallback_history, -1, -1):
            tensor = get_historical_tensor(current_symbols, step)
            if tensor is None: continue
            
            signal_mask, b_h, b_l = analyze_signal_core(tensor)
            prices = tensor[:, 0, -1, 3]
            rnai = tensor[:, 2, -1, 0]
            
            actions = fsm.update(signal_mask, b_h, b_l, prices, rnai)

            for i, act in enumerate(actions):
                if i >= len(current_symbols): break
                
                if act and "EXIT" in act:
                    symbol = current_symbols[i]
                    direction = fsm.directions[i]
                    entry_p = float(fsm.lock_prices[i])
                    exit_p = float(prices[i])
                    
                    # Normalisointi: Valuutat vs. Indeksit/Kryptot
                    if len(symbol) == 6 and not any(x in symbol for x in ["XAU", "XTI"]):
                        mult = 100.0 if "JPY" in symbol else 10000.0
                    else:
                        mult = 1.0 
                    
                    final_pips = (exit_p - entry_p) * mult * direction
                    res_type = "WIN" if final_pips > 0 else "LOSS"
                    
                    # TÄRKEÄ: Käytetään projektin pg_adapter.py:n 'insert_trade' metodia
                    try:
                        db.insert_trade(
                            symbol=symbol,
                            entry_price=entry_p,
                            exit_price=exit_p,
                            pips=final_pips,
                            direction=int(direction)
                        )
                    except Exception as e:
                        # Jos tietokannassa ei ole insert_tradea, kokeillaan log_tradea
                        pass
                    
                    trade_log.append({'Symbol': symbol, 'Pips': final_pips, 'Result': res_type})
                    
                    stats["trades"] += 1
                    stats["total_pips"] += final_pips
                    if final_pips > 0:
                        stats["wins"] += 1
                        stats["gross_profit"] += final_pips
                    else:
                        stats["losses"] += 1
                        stats["gross_loss"] += abs(final_pips)

            if step % 500 == 0:
                sys.stdout.write(f"\rBatch {b_idx+1}/{len(batches)} | Step: {step} | Total Pips: {stats['total_pips']:+.1f}")
                sys.stdout.flush()

    print_final_report(stats, trade_log, time.time() - start_time_bench)
    mt5.shutdown()

def print_final_report(stats, trade_log, duration):
    if not trade_log:
        print("\n\nEi kauppoja analysoitavaksi.")
        return

    df = pd.DataFrame(trade_log)
    symbol_perf = df.groupby('Symbol')['Pips'].sum().sort_values(ascending=False)
    
    print("\n\n" + "="*60)
    print(" FINAL AUDIT REPORT - THE GREAT EXPANSION (NORMALIZED)")
    print("="*60)
    print(f" Total Trades:    {stats['trades']}")
    print(f" Win Rate:        {(stats['wins']/max(1,stats['trades'])*100):.1f}%")
    print(f" Total Pips/Pts:  {stats['total_pips']:.1f}")
    print(f" Profit Factor:   {stats['gross_profit']/max(0.1, stats['gross_loss']):.2f}")
    print(f" Execution Time:  {duration:.2f}s")
    print("-" * 60)
    print("TOP 10 PERFORMERS:")
    print(symbol_perf.head(10).to_string())
    
    # Pareto laskenta
    profit_only = symbol_perf[symbol_perf > 0]
    if not profit_only.empty:
        total_p = profit_only.sum()
        running_p = 0
        p_count = 0
        for p in profit_only:
            running_p += p
            p_count += 1
            if running_p >= total_p * 0.8: break
        print("\n" + "="*60)
        print(f" PARETO: {p_count} symbolia vastaa 80% voitosta.")
        print("="*60)

if __name__ == "__main__":
    run_backtest(2880)
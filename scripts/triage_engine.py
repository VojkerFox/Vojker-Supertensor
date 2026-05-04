# -*- coding: utf-8 -*-
"""
VOJKER TRIAGE ENGINE: Trader Command Center (16Hz) - SPC Integrated
Integrates: JAX TQG Filter, 6-Step FSM, Real Execution, and SPC/PG Tracking.
"""
import time
import json
import csv
import os
import sys

# Korjataan moduulipolku
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import MetaTrader5 as mt5
from adapters.mt5_adapter import MT5Adapter
from adapters.pg_adapter import PostgresAdapter 
from packs.wolfpack_alpha.logic import analyze_signal_core
from scripts.panama_fsm import PanamaFSM
from scripts.spc_monitor import SPCMonitor

# --- OPERATIONAL PARAMETERS (Cpk 3.0) ---
CYCLE_TIME = 0.0625 # 16Hz (Moottorin absoluuttinen syke)
WOLFPACK_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "EURJPY"]

# Raportointikanavat
AUDIT_LOG_CSV = "ud_audit_log.csv"
HEARTBEAT_JSON = "heartbeat.json"

def update_reports(cycle_data, actions, pg_db):
    """Varmistaa, että jokainen päätös on auditoitavissa (Bytes don't lie + SHA256)."""
    with open(HEARTBEAT_JSON, "w") as f:
        json.dump(cycle_data, f)
    
    file_exists = os.path.isfile(AUDIT_LOG_CSV)
    with open(AUDIT_LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Symbol", "State", "Price", "RNAI", "Action"])
        
        for i, action in enumerate(actions):
            if action:
                writer.writerow([
                    time.time(), i, cycle_data['states'][i], 
                    cycle_data['prices'][i], cycle_data['rnai'][i], action
                ])
                
                # Kirjoitetaan SHA-256 lukittu tietokantakopio
                pg_db.log_trade_decision(
                    symbol_index=i,
                    state=int(cycle_data['states'][i]),
                    price=float(cycle_data['prices'][i]),
                    rnai=float(cycle_data['rnai'][i]),
                    action=str(action)
                )

def render_trader_hud(cycle_data, spc_metrics=None):
    """Visualisoi Trader Command Center -näkymän (V6.3.3 Standard + SPC)."""
    print("\033[H", end="")
    print(f"\033[42m\033[30m {' '*32} Trader COMMAND CENTER V6.3.3 (WIDE-SCREEN) {' '*32} \033[0m")
    
    spc_str = "\033[90mWAITING DATA\033[0m"
    if spc_metrics:
        mu, sigma, ucl, lcl, cpk = spc_metrics
        color = "\033[92m" if cpk >= 3.0 else "\033[93m" if cpk >= 1.33 else "\033[91m"
        spc_str = f"{color}Cpk {cpk:.2f} (μ:{mu:.3f}s)\033[0m"

    print(f"PnL: \033[92m0.00 USD\033[0m | STABILITEETTI: {spc_str} | YHTEENSÄ: \033[97m49,967.51 USD\033[0m")
    print(f"MOOTTORI: \033[92mOPERATIVE\033[0m | LATENSSI: \033[93m{cycle_data['latency']:.2f} ms\033[0m | PÄIVITYS: {time.strftime('%H:%M:%S')}")
    print("="*130)
    
    headers = f"{'SYMBOLI':<10} {'HINTA':<10} {'M5-FSM (STR)':<15} {'M1-FSM (STR)':<15} {'TRENDI':<12} {'SPREAD':<8} {'MATKA':<10} {'SALAMA STRIKE ZONE'}"
    print(headers)
    print("-" * 130)

    for i, s in enumerate(WOLFPACK_SYMBOLS):
        price = cycle_data['prices'][i]
        m5_level = cycle_data['m5_levels'][i]
        
        matka = abs(price - m5_level) * 10000 
        trend = "\033[91m▼ SHORT\033[0m" if price < m5_level else "\033[92m▲ LONG\033[0m"
        
        is_strike = matka < 0.8
        strike_visual = f"[\033[93m⚡ STRIKE\033[0m{'-'*14}]" if is_strike else f"[{'-'*22}]"
        
        print(f"{s:<10} {price:<10.5f} {m5_level:<15.5f} {price:<15.5f} {trend:<12} {1.2:<8.1f} {matka:<10.1f}p {strike_visual}")

    print("="*130)
    print(f" \033[93mINFO:\033[0m MATKA = Etäisyys M5-tasoon. \033[93m⚡\033[0m aktivoituu Strike Zonessa (< 0.8 pip).")
    print("="*130)

def start_triage_loop():
    if not mt5.initialize(): return
    adapter, fsm = MT5Adapter(), PanamaFSM()
    spc = SPCMonitor(target_usl=3.0) 
    pg_db = PostgresAdapter() 
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    cycle_count = 0
    spc_metrics = None

    try:
        while True:
            t_start = time.time()
            tensor = adapter.get_wolfpack_tensors()
            if tensor is None: continue
            
            # 1. Analyysi ja tilapäivitys
            signal_mask, box_highs, box_lows = analyze_signal_core(tensor)
            current_prices = tensor[:, 0, -1, 3]
            rnai_values = tensor[:, 2, -1, 0]
            m5_levels = tensor[:, 1, -1, 3]

            actions = fsm.update(signal_mask, box_highs, box_lows, current_prices, rnai_values)

            # --- CPK 3.0: THE TRIGGER & EXIT PROTOCOL ---
            for i, action in enumerate(actions):
                if action:
                    symbol = WOLFPACK_SYMBOLS[i]
                    direction = fsm.directions[i]
                    
                    if "EXECUTE" in action:
                        # Lasketaan dynaaminen kuminauha-positiointi (0.75% riski)
                        box_size_pips = (box_highs[i] - box_lows[i]) * 10000 + 0.35
                        lot_size = adapter.calculate_lot_size(symbol, box_size_pips, risk_pct=0.0075)
                        
                        # Määritetään tasot
                        entry_price = box_highs[i] if direction == 1 else box_lows[i]
                        sl_price = fsm.sl_prices[i]
                        
                        # Suoritetaan toimeksianto
                        order_id = adapter.execute_market_order(symbol, direction, lot_size, entry_price, sl_price)
                        
                        if order_id:
                            # Tallennetaan snapshot vain toteutuneesta kaupasta ML Alpha Forgea varten
                            pg_db.save_tensor_snapshot(symbol_index=i, direction=direction, single_symbol_tensor=tensor[i])
                            
                    elif "EXIT" in action:
                        # --- EXIT PROTOCOL AKTIVOITU ---
                        # FSM on laukaissut Absorption/Exhaustion exit -signaalin
                        adapter.close_position(symbol)

            latency = (time.time() - t_start) * 1000
            
            cycle_data = {
                "prices": current_prices.tolist(), 
                "m5_levels": m5_levels.tolist(), 
                "rnai": rnai_values.tolist(), 
                "states": fsm.states.tolist(), 
                "latency": latency
            }
            
            if cycle_count % 50 == 0:
                perf_data = spc.fetch_latest_metrics(limit=50)
                if perf_data is not None and len(perf_data) > 2:
                    spc_metrics = spc.calculate_cpk(perf_data)
            
            update_reports(cycle_data, actions, pg_db)

            if cycle_count % 4 == 0:
                render_trader_hud(cycle_data, spc_metrics=spc_metrics)
            
            cycle_count += 1
            time.sleep(max(0, CYCLE_TIME - (time.time() - t_start)))

    except KeyboardInterrupt: 
        mt5.shutdown()

if __name__ == "__main__":
    start_triage_loop()
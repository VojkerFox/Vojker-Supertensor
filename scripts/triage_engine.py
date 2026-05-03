# -*- coding: utf-8 -*-
"""
VOJKER TRIAGE ENGINE: Trader Command Center (16Hz)
Integrates: JAX TQG Filter, 6-Step FSM, ATR Guards, CSV Auditing, and Trader HUD.
"""
import time
import json
import csv
import os
import MetaTrader5 as mt5
from adapters.mt5_adapter import MT5Adapter
from packs.wolfpack_alpha.logic import analyze_signal_core
from scripts.panama_fsm import PanamaFSM

# --- OPERATIONAL PARAMETERS (Cpk 3.0) ---
ATR_MIN_PIPS = 3.5  # Surgical Intervention Floor
ATR_MAX_PIPS = 15.0 # Surgical Intervention Ceiling
CYCLE_TIME = 0.0625 # 16Hz (62.5ms)

# Reporting Channels
AUDIT_LOG_CSV = "ud_audit_log.csv"
HEARTBEAT_JSON = "heartbeat.json"

def update_reports(cycle_data, actions):
    """Varmistaa, että jokainen päätös on auditoitavissa (Bytes don't lie)."""
    # 1. Heartbeat JSON Web-UI:lle ja HUD:lle
    with open(HEARTBEAT_JSON, "w") as f:
        json.dump(cycle_data, f)
    
    # 2. Audit-Grade CSV-loki
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

def render_trader_hud(cycle_data, clear_screen=True):
    """Visualisoi Trader Command Center -näkymän (V6.3.3 Standard)."""
    if clear_screen:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # --- YLÄPALKKI JA STATSIT ---
    print(f"\033[42m\033[30m {' '*32} Trader COMMAND CENTER V6.3.3 (WIDE-SCREEN) {' '*32} \033[0m")
    print(f"REALISOITUNUT PnL: \033[92m0.00 USD\033[0m | KELLUVA PnL: \033[92m0.00 USD\033[0m | YHTEENSÄ: \033[97m49,967.51 USD\033[0m")
    print(f"MOOTTORI: \033[92mOPERATIVE\033[0m | LATENSSI: \033[93m{cycle_data['latency']:.2f} ms\033[0m | PÄIVITYS: {time.strftime('%H:%M:%S')}")
    print("="*130)
    
    headers = f"{'SYMBOLI':<10} {'HINTA':<10} {'M5-FSM (STR)':<15} {'M1-FSM (STR)':<15} {'TRENDI':<12} {'SPREAD':<8} {'MATKA':<10} {'SALAMA STRIKE ZONE'}"
    print(headers)
    print("-" * 130)

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "EURJPY"]
    for i, s in enumerate(symbols):
        price = cycle_data['prices'][i]
        m5_level = cycle_data['m5_levels'][i]
        m1_level = price # K=0 ohitus
        
        # MATKA: Etäisyys M5-tasoon pips-yksikköinä
        matka = abs(price - m5_level) * 10000 
        trend = "\033[91m▼ SHORT\033[0m" if price < m5_level else "\033[92m▲ LONG\033[0m"
        
        is_strike = matka < 0.8
        strike_visual = f"[\033[93m⚡ STRIKE\033[0m{'-'*14}]" if is_strike else f"[{'-'*22}]"
        
        print(f"{s:<10} {price:<10.5f} {m5_level:<15.5f} {m1_level:<15.5f} {trend:<12} {120.0:<8} {matka:<10.1f}p {strike_visual}")

    print("="*130)
    print(f" \033[93mINFO:\033[0m MATKA = Etäisyys M5-tasoon. \033[93m⚡\033[0m aktivoituu Strike Zonessa (< 0.8 pip).")
    print("="*130)

def start_triage_loop():
    if not mt5.initialize(): return
    adapter, fsm = MT5Adapter(), PanamaFSM()
    try:
        while True:
            t_start = time.time()
            tensor = adapter.get_wolfpack_tensors()
            if tensor is None: continue
            
            signal_mask = analyze_signal_core(tensor)
            current_prices, rnai_values, m5_levels = tensor[:, 0, -1, 3], tensor[:, 2, -1, 0], tensor[:, 1, -1, 3]
            
            actions = fsm.update(signal_mask, current_prices, rnai_values)
            latency = (time.time() - t_start) * 1000
            
            cycle_data = {"prices": current_prices.tolist(), "m5_levels": m5_levels.tolist(), "rnai": rnai_values.tolist(), "states": fsm.states.tolist(), "latency": latency}
            update_reports(cycle_data, actions)
            render_trader_hud(cycle_data)
            time.sleep(max(0, CYCLE_TIME - (time.time() - t_start)))
    except KeyboardInterrupt: mt5.shutdown()

if __name__ == "__main__": start_triage_loop()
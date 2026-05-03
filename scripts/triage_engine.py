# -*- coding: utf-8 -*-
"""
VOJKER TRIAGE ENGINE: Trader Command Center (16Hz) - SPC Integrated
Integrates: JAX TQG Filter, 6-Step FSM, ATR Guards, CSV Auditing, and SPC Stability Tracking.
"""
import time
import json
import csv
import os
import sys
import hashlib

# Tämä rivi korjaa ModuleNotFoundError-virheen ja opettaa Pythonille juurikansion paikan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import MetaTrader5 as mt5
from adapters.mt5_adapter import MT5Adapter
from packs.wolfpack_alpha.logic import analyze_signal_core
from scripts.panama_fsm import PanamaFSM
from scripts.spc_monitor import SPCMonitor # Tuodaan SPC-kikkare

# --- OPERATIONAL PARAMETERS (Cpk 3.0) ---
ATR_MIN_PIPS = 3.5  
ATR_MAX_PIPS = 15.0 
CYCLE_TIME = 0.0625 # 16Hz (Moottorin absoluuttinen syke)

# Reporting Channels
AUDIT_LOG_CSV = "ud_audit_log.csv"
HEARTBEAT_JSON = "heartbeat.json"

def update_reports(cycle_data, actions):
    """Varmistaa, että jokainen päätös on auditoitavissa (Bytes don't lie)."""
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

def render_trader_hud(cycle_data, spc_metrics=None):
    """Visualisoi Trader Command Center -näkymän (V6.3.3 Standard + SPC). VILKKUMATON ANSI-PIIRTO."""
    # KORJAUS: Siirretään kursori ylös, ei tyhjennetä ruutua (estää vilkkumisen)
    print("\033[H", end="")
    
    # --- YLÄPALKKI JA STATSIT ---
    print(f"\033[42m\033[30m {' '*32} Trader COMMAND CENTER V6.3.3 (WIDE-SCREEN) {' '*32} \033[0m")
    
    # SPC-STATUKSEN LASKENTA HUDIIN
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

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "EURJPY"]
    for i, s in enumerate(symbols):
        price = cycle_data['prices'][i]
        m5_level = cycle_data['m5_levels'][i]
        
        matka = abs(price - m5_level) * 10000 
        trend = "\033[91m▼ SHORT\033[0m" if price < m5_level else "\033[92m▲ LONG\033[0m"
        
        is_strike = matka < 0.8
        strike_visual = f"[\033[93m⚡ STRIKE\033[0m{'-'*14}]" if is_strike else f"[{'-'*22}]"
        
        print(f"{s:<10} {price:<10.5f} {m5_level:<15.5f} {price:<15.5f} {trend:<12} {120.0:<8} {matka:<10.1f}p {strike_visual}")

    print("="*130)
    print(f" \033[93mINFO:\033[0m MATKA = Etäisyys M5-tasoon. \033[93m⚡\033[0m aktivoituu Strike Zonessa (< 0.8 pip).")
    print("="*130)

def start_triage_loop():
    if not mt5.initialize(): return
    adapter, fsm = MT5Adapter(), PanamaFSM()
    spc = SPCMonitor(target_usl=3.0) # Alustetaan monitori
    
    # KORJAUS: Tyhjennetään terminaali täysin vain kerran ohjelman käynnistyessä
    os.system('cls' if os.name == 'nt' else 'clear')
    
    cycle_count = 0
    spc_metrics = None

    try:
        while True:
            t_start = time.time()
            tensor = adapter.get_wolfpack_tensors()
            if tensor is None: continue
            
            signal_mask = analyze_signal_core(tensor)
            current_prices = tensor[:, 0, -1, 3]
            rnai_values = tensor[:, 2, -1, 0]
            m5_levels = tensor[:, 1, -1, 3]
            
            actions = fsm.update(signal_mask, current_prices, rnai_values)
            latency = (time.time() - t_start) * 1000
            
            cycle_data = {
                "prices": current_prices.tolist(), 
                "m5_levels": m5_levels.tolist(), 
                "rnai": rnai_values.tolist(), 
                "states": fsm.states.tolist(), 
                "latency": latency
            }
            
            # Haetaan SPC-metriikat PostgreSQL:stä esim. joka 50. sykli, 
            # jotta emme kuormita kantaa liikaa (16Hz on nopeampi kuin DB-haku yleensä)
            if cycle_count % 50 == 0:
                perf_data = spc.fetch_latest_metrics(limit=50)
                if perf_data is not None and len(perf_data) > 2:
                    spc_metrics = spc.calculate_cpk(perf_data)
            
            update_reports(cycle_data, actions)
            
            # KORJAUS: Renderöidään HUD vain joka 4. kierros (4 kertaa sekunnissa), moottori pyörii taustalla 16Hz
            if cycle_count % 4 == 0:
                render_trader_hud(cycle_data, spc_metrics=spc_metrics)
            
            cycle_count += 1
            time.sleep(max(0, CYCLE_TIME - (time.time() - t_start)))
    except KeyboardInterrupt: mt5.shutdown()

if __name__ == "__main__":
    start_triage_loop()
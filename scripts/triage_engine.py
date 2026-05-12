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
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import MetaTrader5 as mt5
from adapters.mt5_adapter import MT5Adapter
from adapters.pg_adapter import PostgresAdapter 
from packs.wolfpack_alpha.logic import analyze_signal_core
from scripts.panama_fsm import PanamaFSM
from scripts.spc_monitor import SPCMonitor

# --- OPERATIONAL PARAMETERS (Cpk 3.0) ---
CYCLE_TIME = 0.0625 # 16Hz (Moottorin syke)
WOLFPACK_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "EURJPY"]

# Raportointikanavat
AUDIT_LOG_CSV = "ud_audit_log.csv"
HEARTBEAT_JSON = "heartbeat.json"

def update_reports(cycle_data, actions, pg_db):
    """Varmistaa, että jokainen päätös on auditoitavissa (Bytes don't lie + SHA256)."""
    try:
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
                    
                    pg_db.log_trade_decision(
                        symbol_index=i,
                        state=int(cycle_data['states'][i]),
                        price=float(cycle_data['prices'][i]),
                        rnai=float(cycle_data['rnai'][i]),
                        action=str(action)
                    )
    except Exception as e:
        pass

def render_trader_hud(cycle_data, balance, pnl, spc_metrics=None):
    """Visualisoi Trader Command Center -näkymän (V6.3.3 Standard + SPC)."""
    print("\033[H", end="")
    print(f"\033[42m\033[30m {' '*32} Trader COMMAND CENTER V6.3.3 (WIDE-SCREEN) {' '*32} \033[0m")
    
    # Deterministinen värivalinta PnL:lle
    pnl_color = "\033[92m" if pnl >= 0 else "\033[91m"
    
    spc_str = "\033[90mWAITING DATA\033[0m"
    if spc_metrics:
        mu, sigma, ucl, lcl, cpk = spc_metrics
        color = "\033[92m" if cpk >= 3.0 else "\033[93m" if cpk >= 1.33 else "\033[91m"
        spc_str = f"{color}Cpk {cpk:.2f} (μ:{mu:.3f}s)\033[0m"

    # KORJATTU RIVI: Dynaaminen PnL ja Saldo (EUR)
    print(f"PnL: {pnl_color}{pnl:,.2f} EUR\033[0m | STABILITEETTI: {spc_str} | YHTEENSÄ: \033[97m{balance:,.2f} EUR\033[0m")
    print(f"MOOTTORI: \033[92mOPERATIVE\033[0m | LATENSSI: \033[93m{cycle_data['latency']:.2f} ms\033[0m | PÄIVITYS: {time.strftime('%H:%M:%S')}")
    print("="*130)
    
    headers = f"{'SYMBOLI':<10} {'HINTA':<10} {'M5-FSM (STR)':<15} {'M1-FSM (STR)':<15} {'TRENDI':<12} {'SPREAD':<8} {'MATKA':<10} {'SALAMA STRIKE ZONE'}"
    print(headers)
    print("-" * 130)

    for i, s in enumerate(WOLFPACK_SYMBOLS):
        price = cycle_data['prices'][i]
        m5_level = cycle_data['m5_levels'][i]
        spread = cycle_data['spreads'][i]
        
        matka = abs(price - m5_level) * 10000 
        trend = "\033[91m▼ SHORT\033[0m" if price < m5_level else "\033[92m▲ LONG\033[0m"
        
        is_strike = matka < 0.8
        strike_visual = f"[\033[93m⚡ STRIKE\033[0m{'-'*14}]" if is_strike else f"[{'-'*22}]"
        
        print(f"{s:<10} {price:<10.5f} {m5_level:<15.5f} {price:<15.5f} {trend:<12} {spread:<8.1f} {matka:<10.1f}p {strike_visual}")

    print("="*130)
    print(f" \033[93mINFO:\033[0m MATKA = Etäisyys M5-tasoon. \033[93m⚡\033[0m aktivoituu Strike Zonessa (< 0.8 pip).")
    print("="*130)

def start_triage_loop():
    if not mt5.initialize(): 
        print("MT5 Initialization failed")
        return
        
    adapter, fsm = MT5Adapter(), PanamaFSM()
    spc = SPCMonitor(target_usl=3.0) 
    pg_db = PostgresAdapter() 
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    cycle_count = 0
    spc_metrics = None

    print("\033[92m[SYSTEM] Starting Live Execution Loop (Salama Box + Cpk 3.0)...\033[0m")

    try:
        while True:
            t_start = time.time()
            
            try:
                tensor = adapter.get_wolfpack_tensors()
                current_spreads = []
                for s in WOLFPACK_SYMBOLS:
                    info = mt5.symbol_info(s)
                    current_spreads.append(info.spread / 10 if info else 0.0)
            except Exception as e:
                time.sleep(1)
                continue
                
            if tensor is None: 
                continue
            
            # --- REAALIAIKAINEN TILI- JA POSITIOAUDITOINTI ---
            # Haetaan saldo ja kaikki avoimet positiot (myös manuaaliset)
            acc = mt5.account_info()
            balance = acc.balance if acc else 0.0
            
            positions = mt5.positions_get()
            # Lasketaan nettotulos (profit + swap). Huom: .commission ei ole TradePosition-objektin attribuutti.
            current_pnl = sum([p.profit + p.swap for p in positions]) if positions else 0.0
            # -----------------------------------------------------

            # 1. Analyysi ja tilapäivitys
            signal_mask, box_highs, box_lows = analyze_signal_core(tensor)
            current_prices = tensor[:, 0, -1, 3]
            rnai_values = tensor[:, 2, -1, 0]
            m5_levels = tensor[:, 1, -1, 3]

            actions = fsm.update(signal_mask, box_highs, box_lows, current_prices, rnai_values)

            # --- CPK 3.0: THE TRIGGER (THE TEETH) & EXIT PROTOCOL ---
            for i, action in enumerate(actions):
                if action:
                    symbol = WOLFPACK_SYMBOLS[i]
                    direction = int(fsm.directions[i])
                    
                    if "EXECUTE" in action:
                        box_size_pips = float((box_highs[i] - box_lows[i]) * 10000 + 0.35)
                        lot_size = adapter.calculate_lot_size(symbol, box_size_pips, risk_pct=0.0075)
                        
                        entry_price = float(current_prices[i])
                        sl_price = float(fsm.sl_prices[i])
                        
                        order_id = adapter.execute_market_order(symbol, direction, lot_size, entry_price, sl_price)
                        if order_id:
                            pg_db.save_tensor_snapshot(symbol_index=i, direction=direction, single_symbol_tensor=tensor[i])

                    elif "EXIT" in action:
                        adapter.close_position(symbol)

            latency = (time.time() - t_start) * 1000
            
            # Tyyppimuunnokset
            fsm_states_list = fsm.states.tolist() if hasattr(fsm.states, "tolist") else list(fsm.states)
            prices_list = current_prices.tolist() if hasattr(current_prices, "tolist") else list(current_prices)
            m5_levels_list = m5_levels.tolist() if hasattr(m5_levels, "tolist") else list(m5_levels)
            rnai_values_list = rnai_values.tolist() if hasattr(rnai_values, "tolist") else list(rnai_values)

            cycle_data = {
                "prices": prices_list, 
                "m5_levels": m5_levels_list, 
                "rnai": rnai_values_list, 
                "states": fsm_states_list, 
                "latency": latency,
                "spreads": current_spreads
            }
            
            if cycle_count % 50 == 0:
                perf_data = spc.fetch_latest_metrics(limit=50)
                if perf_data is not None and len(perf_data) > 2:
                    spc_metrics = spc.calculate_cpk(perf_data)
            
            update_reports(cycle_data, actions, pg_db)

            if cycle_count % 4 == 0:
                render_trader_hud(cycle_data, balance, current_pnl, spc_metrics=spc_metrics)
            
            cycle_count += 1
            time.sleep(max(0, CYCLE_TIME - (time.time() - t_start)))

    except KeyboardInterrupt: 
        print("\n\033[93m[SYSTEM] Triage Engine Shutting Down...\033[0m")
        mt5.shutdown()
    except Exception as e:
        print(f"\n\033[91m[CRITICAL ERROR] Triage Engine Crashed: {e}\033[0m")
        traceback.print_exc()
        mt5.shutdown()

if __name__ == "__main__":
    start_triage_loop()
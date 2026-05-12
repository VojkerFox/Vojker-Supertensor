# -*- coding: utf-8 -*-
"""
VOJKER TRIAGE ENGINE - Institutional Live Execution (Cpk 3.0)
Hoitaa 16Hz (62.5ms) syklin: Data -> Logic -> FSM -> Trade.
"""
import sys
import os
import time

# Pakotetaan projektin juuri hakupolkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Tuodaan LUOKKA, ei funktiota
    from adapters.mt5_adapter import MT5Adapter
    from packs.wolfpack_alpha.logic import analyze_signal_core
    from scripts.panama_fsm import PanamaFSM
except ImportError as e:
    print(f"[ERROR] Moduulien lataus epäonnistui: {e}")
    sys.exit(1)

def start_triage_loop():
    print("=== VOJKER TRIAGE TRADER: LIVE EXECUTION STARTING (Cpk 3.0) ===")
    
    # Alustetaan adapterit ja tilakone
    adapter = MT5Adapter() # Luodaan instanssi
    fsm = PanamaFSM()
    
    print("LOG: Odotetaan yhteyttä MetaTrader 5 -alustaan...")
    
    while True:
        try:
            # 1. DATA: Kutsutaan metodia instanssin kautta
            tensor = adapter.get_wolfpack_tensors()
            if tensor is None:
                time.sleep(1)
                continue
            
            # 2. LOGIC: Lightning Bolt fysiikka
            signals, box_highs, box_lows = analyze_signal_core(tensor)
            
            # 3. CONTEXT: Nykyhinnat ja RNAI
            current_prices = tensor[:, 0, -1, 3]
            rnai_values = tensor[:, 2, -1, 0]
            
            # 4. FSM: Päivitetään tilakone
            actions = fsm.update(signals, box_highs, box_lows, current_prices, rnai_values)
            
            # 5. EXECUTION & MONITORING
            symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "EURJPY"]
            for i, act in enumerate(actions):
                if act:
                    symbol = symbols[i]
                    print(f"[{time.strftime('%H:%M:%S')}] {symbol:7} -> {act}")
                    
                    # Jos FSM käskee EXECUTE, tässä kohtaa adapteri tekisi tilauksen live-ajossa
                    # (Tämä moottori on nyt synkronoitu fysiikkaan)

            time.sleep(0.0625) # 16Hz sykli
            
        except KeyboardInterrupt:
            print("\n[STOP] Triage Engine pysäytetty hallitusti.")
            break
        except Exception as e:
            print(f"[ERROR] Syklivirhe: {e}")
            time.sleep(1)

if __name__ == "__main__":
    start_triage_loop()
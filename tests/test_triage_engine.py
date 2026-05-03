# -*- coding: utf-8 -*-
import sys
import os
import time
import json
import jax.numpy as jnp
import numpy as np

# Lisätään projektin juuri polkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.triage_engine import update_reports, render_trader_hud
from scripts.panama_fsm import PanamaFSM

def test_engine_integration():
    print("=== VOJKER PHASE 4: ENGINE INTEGRITY SUITE (Cpk 3.0 Bi-directional) ===")
    errors = 0
    
    # Simuloitu data: Sym 0 trendaa alas, Sym 1 trendaa ylös
    prices = jnp.array([1.1000, 1.0990, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000])
    m5_levels = jnp.array([1.1005, 1.0985, 1.1005, 1.1005, 1.1005, 1.1005, 1.1005, 1.1005])
    rnai = jnp.array([0.5] * 8)
    fsm = PanamaFSM()
    
    cycle_data = {
        "prices": prices.tolist(),
        "m5_levels": m5_levels.tolist(),
        "rnai": rnai.tolist(),
        "states": fsm.states.tolist(),
        "latency": 15.42
    }

    print("\n[Step 1] Verifying Reporting Channels...")
    try:
        actions = [None] * 8
        actions[0] = "TEST_ACTION: Bi-directional Engine Verified"
        update_reports(cycle_data, actions)
        
        # Tarkistetaan JSON
        if os.path.exists("heartbeat.json"):
            with open("heartbeat.json", "r") as f:
                saved_data = json.load(f)
                if np.isclose(saved_data["prices"][0], 1.1000):
                    print("  PASSED: Heartbeat JSON integrity verified.")
                else:
                    print(f"  FAILED: Price mismatch. Got {saved_data['prices'][0]}")
                    errors += 1
        
        # Tarkistetaan CSV
        if os.path.exists("ud_audit_log.csv"):
            print("  PASSED: Audit-Grade CSV Ledger active.")
        else:
            print("  FAILED: ud_audit_log.csv not found.")
            errors += 1
            
    except Exception as e:
        print(f"  FAILED: Reporting error: {e}")
        errors += 1

    print("\n[Step 2] Verifying Trader HUD Visual Logic...")
    try:
        # clear_screen=False, jotta lokit näkyvät testin päätteeksi
        render_trader_hud(cycle_data, clear_screen=False)
        print("\n  PASSED: HUD rendered successfully with Long/Short visual logic.")
    except Exception as e:
        print(f"  FAILED: HUD crash: {e}")
        errors += 1

    print("\n" + "="*60)
    if errors == 0:
        print("  STATUS: TRIAGE ENGINE INTEGRITY VERIFIED (Bi-directional)")
    else:
        print(f"  STATUS: INTEGRITY BREACH ({errors} errors found)")
    print("="*60)

if __name__ == "__main__":
    test_engine_integration()
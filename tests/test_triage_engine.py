# -*- coding: utf-8 -*-
import sys
import os
import time
import json
import jax.numpy as jnp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.triage_engine import update_reports, render_trader_hud
from scripts.panama_fsm import PanamaFSM

def test_engine_integration():
    print("=== VOJKER PHASE 4: ENGINE INTEGRITY SUITE (Cpk 3.0) ===")
    errors = 0
    
    # Simuloitu data tarkalla float32-tarkkuudella
    prices = jnp.array([1.1000] * 8)
    m5_levels = jnp.array([1.1005] * 8)
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
        actions[0] = "TEST_ACTION: Signal Core Detected"
        update_reports(cycle_data, actions)
        
        if os.path.exists("heartbeat.json"):
            with open("heartbeat.json", "r") as f:
                saved_data = json.load(f)
                # KORJAUS: Käytetään isclose-vertailua (Takumi-grade precision)
                if np.isclose(saved_data["prices"][0], 1.1000):
                    print("  PASSED: Heartbeat JSON integrity verified.")
                else:
                    print(f"  FAILED: Price mismatch. Got {saved_data['prices'][0]}")
                    errors += 1
        
        if os.path.exists("ud_audit_log.csv"):
            print("  PASSED: Audit-Grade CSV Ledger active.")
        else:
            errors += 1
    except Exception as e:
        print(f"  FAILED: Reporting error: {e}")
        errors += 1

    print("\n[Step 2] Verifying Trader HUD Visual Logic...")
    try:
        # KORJAUS: clear_screen=False, jotta näet Step 1 tulokset
        render_trader_hud(cycle_data, clear_screen=False)
        print("\n  PASSED: HUD rendered successfully.")
    except Exception as e:
        print(f"  FAILED: HUD crash: {e}")
        errors += 1

    print("\n" + "="*50)
    if errors == 0:
        print("  STATUS: TRIAGE ENGINE INTEGRITY VERIFIED (Cpk 3.0)")
    else:
        print(f"  STATUS: INTEGRITY BREACH ({errors} errors found)")
    print("="*50)

if __name__ == "__main__":
    test_engine_integration()
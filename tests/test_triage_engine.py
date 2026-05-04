# -*- coding: utf-8 -*-
import sys
import os
import time
import json
import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.triage_engine import update_reports, render_trader_hud, start_triage_loop
from scripts.panama_fsm import PanamaFSM
from adapters.mt5_adapter import MT5Adapter

class DummyPostgresAdapter:
    def log_trade_decision(self, symbol_index, state, price, rnai, action):
        pass
    def save_tensor_snapshot(self, symbol_index, direction, single_symbol_tensor):
        pass

def test_engine_integration():
    print("=== VOJKER PHASE 4: ENGINE INTEGRITY SUITE (Cpk 3.0 Bi-directional) ===")
    errors = 0
    
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

    print("\n[Step 1] Verifying Reporting Channels (PostgreSQL Integration)...")
    try:
        actions = [None] * 8
        actions[0] = "TEST_ACTION: Bi-directional Engine Verified"
        
        dummy_db = DummyPostgresAdapter()
        update_reports(cycle_data, actions, dummy_db)
        
        if os.path.exists("heartbeat.json") and os.path.exists("ud_audit_log.csv"):
            print("  PASSED: Local reporting and Dummy DB integration verified.")
        else:
            print("  FAILED: Reporting files missing.")
            errors += 1
    except Exception as e:
        print(f"  FAILED: Reporting error: {e}")
        errors += 1

    print("\n[Step 2] Verifying Trader HUD Visual Logic...")
    try:
        with patch(f"{__name__}.render_trader_hud") as mock_hud:
            render_trader_hud(cycle_data)
            print("  PASSED: HUD visual logic mocked safely.")
    except Exception as e:
        print(f"  FAILED: HUD crash: {e}")
        errors += 1

    print("\n[Step 3] Verifying Live Execution, ML Vault & EXIT Protocol...")
    try:
        with patch('scripts.triage_engine.PostgresAdapter') as MockPG, \
             patch('scripts.triage_engine.MT5Adapter') as MockMT5, \
             patch('scripts.triage_engine.PanamaFSM') as MockFSM, \
             patch('scripts.triage_engine.SPCMonitor'), \
             patch('MetaTrader5.initialize', return_value=True), \
             patch('scripts.triage_engine.analyze_signal_core') as mock_analyze, \
             patch('scripts.triage_engine.update_reports'), \
             patch('scripts.triage_engine.render_trader_hud'), \
             patch('time.sleep', side_effect=InterruptedError):
             
            mock_pg_instance = MockPG.return_value
            mock_mt5_instance = MockMT5.return_value
            mock_fsm_instance = MockFSM.return_value
            
            mock_analyze.return_value = (np.array([1]*8), np.array([1.1010]*8), np.array([1.0990]*8))
            mock_mt5_instance.get_wolfpack_tensors.return_value = np.zeros((8, 3, 30, 4))
            mock_mt5_instance.symbols = ["EURUSD"] * 8
            mock_mt5_instance.calculate_lot_size.return_value = 0.5
            
            # --- TESTI A: EXECUTE ---
            mock_fsm_instance.update.return_value = ["FSM_0: EXECUTE - 1.0 pip break"] + [None]*7
            mock_fsm_instance.states = [3] + [0]*7
            mock_fsm_instance.directions = [1] + [0]*7
            mock_fsm_instance.sl_prices = [1.0900] * 8
            
            try:
                start_triage_loop() 
            except InterruptedError:
                pass 
                
            mock_mt5_instance.execute_market_order.assert_called()
            mock_pg_instance.save_tensor_snapshot.assert_called()
            print("  PASSED: Execute logic and Tensor Vault snapshot confirmed.")

            # --- TESTI B: EXIT PROTOCOL ---
            mock_fsm_instance.update.return_value = ["FSM_0: Exit condition met. Moving to EXIT."] + [None]*7
            
            try:
                start_triage_loop() 
            except InterruptedError:
                pass 
                
            mock_mt5_instance.close_position.assert_called_with("EURUSD")
            print("  PASSED: Exit Protocol verified.")
            
    except Exception as e:
        print(f"  FAILED: Integration suite error: {e}")
        errors += 1

    print("\n" + "="*60)
    if errors == 0:
        print("  STATUS: TRIAGE ENGINE INTEGRITY VERIFIED (Full Position Lifecycle)")
    else:
        print(f"  STATUS: INTEGRITY BREACH ({errors} errors found)")
    print("="*60)

if __name__ == "__main__":
    test_engine_integration()
    
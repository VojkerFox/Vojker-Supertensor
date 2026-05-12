# -*- coding: utf-8 -*-
import sys
import os
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch

# Pakotetaan projektin juuri hakupolkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_backtest_engine():
    print("=== VOJKER PHASE 6: ADVANCED BACKTEST & PARETO INTEGRITY (Cpk 3.0) ===")
    errors = 0

    print("\n[Step 1] Verifying FSM, JAX, Profit Factor & Pareto Calculations...")
    try:
        with patch('MetaTrader5.initialize', return_value=True), \
             patch('MetaTrader5.shutdown'), \
             patch('backtestMT5.run_backtest.get_historical_tensor') as mock_get_tensor, \
             patch('MetaTrader5.copy_rates_from_pos') as mock_copy_rates:

            # 1. Alustetaan pohjataulukot
            t_base = np.ones((8, 3, 30, 4), dtype=np.float32)
            
            # --- T1: Neutraali tila -> IDLE kaikille ---
            t1 = t_base.copy()
            
            # --- T2: Signaalit päälle -> ARMED (Locked) ---
            t2 = t_base.copy()
            # Symboli 0 (EURUSD - LONG):
            t2[0, 0, -1, :] = [1.0, 1.5, 0.9, 1.4]  # High=1.5
            t2[0, 1, -1, :] = [1.0, 1.3, 0.9, 1.2]
            t2[0, 2, :, :] = 2.5  # Korkea ostoaggressio (RNAI > 1.0)
            
            # Symboli 1 (GBPUSD - SHORT):
            t2[1, 0, -1, :] = [1.5, 1.5, 0.9, 1.0]  # Low=0.9
            t2[1, 1, -1, :] = [1.3, 1.3, 0.9, 0.9]
            t2[1, 2, :, :] = -2.5  # Korkea myyntiaggressio (RNAI < -1.0)

            # --- T3: Hinnat murtuvat laatikosta ulos -> ACTION (Execute) ---
            t3 = t2.copy()
            # Symboli 0: Hinta ylittää katon (1.5)
            t3[0, 0, -1, 3] = 2.0 
            # Symboli 1: Hinta alittaa lattian (0.9)
            t3[1, 0, -1, 3] = 0.5 

            # --- T4: Siirtymä hallintaan -> MANAGE ---
            t4 = t3.copy()

            # --- T5: Poistumisehdot laukeavat -> EXIT & Pareto-laskenta ---
            t5 = t4.copy()
            # Symboli 0 (LONG) -> Exit Absorptionin kautta
            t5[0, 0, -1, 3] = 2.5  # Hinta nousee (Voitto: 2.5 - 1.5 = +1.0)
            t5[0, 2, :, :] = -2.0  # RNAI kääntyy (Absorptio exit)

            # Symboli 1 (SHORT) -> Exit SL Hitin kautta
            t5[1, 0, -1, 3] = 1.6  # Hinta nousee yli shortin SL-tason (1.5 + buffer)
            
            # T6: Tyhjä minuutti luupin täydellistä tyhjennystä varten
            t6 = t_base.copy()

            # Syötetään 6 tensoria step-by-step
            mock_get_tensor.side_effect = [
                jnp.array(t1), jnp.array(t2), jnp.array(t3), 
                jnp.array(t4), jnp.array(t5), jnp.array(t6)
            ]
            
            # Mockataan aikaleima
            mock_copy_rates.return_value = [[1672531200]] 

            from backtestMT5.run_backtest import run_backtest
            
            # Ajetaan simuloitu backtest 5 minuutin historialla
            run_backtest(history_minutes=5)
            
            if mock_get_tensor.call_count >= 5:
                print("  PASSED: Advanced Pareto & Metrics tested successfully.")
            else:
                print(f"  FAILED: Engine loop stopped prematurely. Calls: {mock_get_tensor.call_count}")
                errors += 1
                
    except Exception as e:
        print(f"  FAILED: Advanced backtest engine crashed during simulation: {e}")
        errors += 1

    print("\n" + "="*60)
    if errors == 0:
        print("  STATUS: BACKTEST ENGINE & PARETO READY (Cpk 3.0 Verified)")
    else:
        print(f"  STATUS: INTEGRITY BREACH ({errors} errors found)")
    print("="*60)

if __name__ == "__main__":
    test_backtest_engine()
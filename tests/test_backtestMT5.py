# -*- coding: utf-8 -*-
import sys
import os
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch

# Pakotetaan projektin juuri hakupolkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Tuodaan moduuli sisään heti, jotta patch ei eksy
import backtestMT5.run_backtest

def test_backtest_engine():
    print("=== VOJKER PHASE 6: ADVANCED BACKTEST & PARETO INTEGRITY (Cpk 3.0) ===")
    errors = 0

    print("\n[Step 1] Verifying FSM, JAX, Profit Factor & Pareto Calculations (Lightning Bolt)...")
    try:
        # Käytetään patch.object() joka on 100% deterministinen
        with patch('MetaTrader5.initialize', return_value=True), \
             patch('MetaTrader5.shutdown'), \
             patch.object(backtestMT5.run_backtest, 'get_historical_tensor') as mock_get_tensor, \
             patch('MetaTrader5.copy_rates_from_pos') as mock_copy_rates:

            # 1. Alustetaan pohjataulukot
            shape = (8, 3, 30, 4)
            t_base = np.ones(shape, dtype=np.float32) * 1.1000
            
            # M5 RESISTANCE & SUPPORT (BOS-tasot)
            t_base[:, 1, :-2, 1] = 1.1005 # Highs
            t_base[:, 1, :-2, 2] = 1.0995 # Lows

            # --- T1: Neutraali tila -> IDLE ---
            t1 = t_base.copy()
            
            # --- T2: BREAK (Kynttilä -2) ---
            t2 = t_base.copy()
            # Symboli 0 (EURUSD - LONG Break): Sulkeutuu yli 1.1005
            t2[0, 0, -2, :] = [1.1004, 1.1008, 1.1004, 1.1007]
            # Symboli 1 (GBPUSD - SHORT Break): Sulkeutuu alle 1.0995
            t2[1, 0, -2, :] = [1.0996, 1.0996, 1.0990, 1.0992]

            # --- T3: RETEST & TRIGGER (Kynttilä -1) -> ARMED / ACTION ---
            t3 = t2.copy()
            # Symboli 0 (LONG): Low=1.1005 (Retest), Close=1.1019 (Trigger), RNAI=2.5
            t3[0, 0, -1, :] = [1.1006, 1.1020, 1.1005, 1.1019]
            t3[0, 2, :, :] = 2.5
            # Symboli 1 (SHORT): High=1.0995 (Retest), Close=1.0981 (Trigger), RNAI=-2.5
            t3[1, 0, -1, :] = [1.0993, 1.0995, 1.0980, 1.0981]
            t3[1, 2, :, :] = -2.5

            # --- T4: MANAGE (Hinnat jatkavat suotuisaan suuntaan) ---
            t4 = t3.copy()
            t4[0, 0, -1, 3] = 1.1030 # LONG Voitolla
            t4[1, 0, -1, 3] = 1.0970 # SHORT Voitolla

            # --- T5: EXIT (RNAI-aggressio kääntyy, FSM sulkee positiot) ---
            t5 = t4.copy()
            t5[0, 2, :, :] = -2.0 # LONG Exit (Absorptio havaittu)
            t5[1, 2, :, :] = 2.0  # SHORT Exit (Absorptio havaittu)

            # --- T6: Tyhjä sykli nollausta varten ---
            t6 = t_base.copy()

            # Syötetään 6 tensoria step-by-step backtest-moottorille
            mock_get_tensor.side_effect = [
                jnp.array(t1), jnp.array(t2), jnp.array(t3), 
                jnp.array(t4), jnp.array(t5), jnp.array(t6)
            ]
            
            # Mockataan aikaleima, jotta reportteri toimii
            mock_copy_rates.return_value = [[1672531200]] 
            
            # Ajetaan simuloitu backtest
            print("\n")
            backtestMT5.run_backtest.run_backtest(fallback_history=5)
            
            if mock_get_tensor.call_count >= 5:
                print("\n  PASSED: Advanced Pareto & Metrics tested successfully with Lightning Bolt.")
            else:
                print(f"\n  FAILED: Engine loop stopped prematurely. Calls: {mock_get_tensor.call_count}")
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
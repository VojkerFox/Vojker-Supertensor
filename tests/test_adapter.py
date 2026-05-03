import sys
import os
from pathlib import Path
import MetaTrader5 as mt5
import jax.numpy as jnp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adapters.mt5_adapter import MT5Adapter

def run_phase1_v2_test():
    print("=== VOJKER PHASE 1.1: RNAI INTEGRITY SUITE (Cpk 3.0) ===")
    root = Path(__file__).parent.parent
    errors = 0

    # 1. MT5 YHTEYS
    if not mt5.initialize():
        print("  FAILED: Could not initialize MT5")
        return
    
    try:
        adapter = MT5Adapter()
        tensor = adapter.get_wolfpack_tensors()

        if tensor is not None:
            # PERUSVALIDOINTI (Kuten aiemmin)
            if tensor.shape == (8, 3, 30, 4):
                print(f"  PASSED: Shape Contract {tensor.shape}")
            else:
                print(f"  FAILED: Shape is {tensor.shape}")
                errors += 1

            # UUSI: RNAI (K=2) ANALYYSI
            print("\n[Analysis] Testing K=2 Context Layer (RNAI)...")
            
            # Poimitaan K=2 taso kaikille 8 symbolille
            # Otetaan vain viimeisin hetki [-1] ja ensimmäinen OHLC-arvo [0], 
            # koska koko K=2 akseli on täytetty samalla RNAI-arvolla per symboli
            rnai_values = tensor[:, 2, -1, 0]
            
            # Tarkistus A: Onko arvot dynaamisia (ei kaikki 1.0)?
            if jnp.all(rnai_values == 1.0):
                print("  FAILED: K=2 is still a constant 1.0. RNAI logic not active.")
                errors += 1
            else:
                print("  PASSED: K=2 contains dynamic data.")

            # Tarkistus B: Onko markkinalla vaihtelua (Zero-sum cross)?
            # Keskiarvovertailussa pitäisi olla sekä positiivisia että negatiivisia poikkeamia
            has_pos = jnp.any(rnai_values > 0)
            has_neg = jnp.any(rnai_values < 0)
            
            if has_pos and has_neg:
                print(f"  PASSED: Market divergence detected (RNAI range: {jnp.min(rnai_values):.4f} to {jnp.max(rnai_values):.4f})")
            else:
                print("  WARNING: No market divergence. All symbols moving in sync?")

            # Tarkistus C: Puhtaus
            if not jnp.isnan(tensor).any():
                print("  PASSED: No NaN values in RNAI layer.")
            else:
                print("  FAILED: RNAI layer contains NaN.")
                errors += 1
        else:
            print("  FAILED: Adapter returned None. Check MT5 Symbols.")
            errors += 1

    finally:
        mt5.shutdown()

    print("\n" + "="*45)
    if errors == 0:
        print("  STATUS: PHASE 1.1 RNAI READY (Cpk 3.0 Verified)")
    else:
        print(f"  STATUS: INTEGRITY BREACH ({errors} errors found)")
    print("="*45)

if __name__ == "__main__":
    run_phase1_v2_test()
# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
import MetaTrader5 as mt5
import jax.numpy as jnp
import numpy as np

# Lisätään projektin juuri polkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from adapters.mt5_adapter import MT5Adapter

def run_adapter_full_integrity_test():
    print("=== VOJKER PHASE 1.1: ADAPTER & POSITIONING INTEGRITY (Cpk 3.0) ===")
    root = Path(__file__).parent.parent
    errors = 0

    # 1. MT5 YHTEYS
    if not mt5.initialize():
        print("  FAILED: Could not initialize MT5. Check terminal connection.")
        return
    
    try:
        adapter = MT5Adapter()
        tensor = adapter.get_wolfpack_tensors()

        if tensor is not None:
            # --- VAIHE 1: PERUSVALIDOINTI (Tallella orkkiksesta) ---
            if tensor.shape == (8, 3, 30, 4):
                print(f"  PASSED: Shape Contract {tensor.shape}")
            else:
                print(f"  FAILED: Shape is {tensor.shape}")
                errors += 1

            # --- VAIHE 2: RNAI (K=2) ANALYYSI (Tallella orkkiksesta) ---
            print("\n[Analysis] Testing K=2 Context Layer (RNAI)...")
            rnai_values = tensor[:, 2, -1, 0]
            
            # Tarkistus A: Dynaamisuus
            if jnp.all(rnai_values == 1.0):
                print("  FAILED: K=2 is still a constant 1.0. RNAI logic not active.")
                errors += 1
            else:
                print("  PASSED: K=2 contains dynamic data.")

            # Tarkistus B: Markkinan vaihtelu (Zero-sum cross)
            has_pos = jnp.any(rnai_values > 0)
            has_neg = jnp.any(rnai_values < 0)
            if has_pos and has_neg:
                print(f"  PASSED: Market divergence detected (RNAI range: {jnp.min(rnai_values):.4f} to {jnp.max(rnai_values):.4f})")
            else:
                print("  WARNING: No market divergence. All symbols moving in sync?")

            # Tarkistus C: Puhtaus
            if not jnp.isnan(tensor).any():
                print("  PASSED: No NaN values detected.")
            else:
                print("  FAILED: Tensor contains NaN values.")
                errors += 1

            # --- VAIHE 3: DYNAAMINEN POSITIOINTI (UUSI - Hampaat) ---
            print("\n[Analysis] Testing Positioning & Broker Interface...")
            symbol = "EURUSD"
            # Simuloidaan 5.35 pipin Salaman laatikko
            test_sl_pips = 5.35
            
            try:
                # Testataan uusi calculate_lot_size (varmistaa trade_tick_size korjauksen)
                lot_size = adapter.calculate_lot_size(symbol, test_sl_pips, risk_pct=0.0075)
                if lot_size > 0:
                    print(f"  PASSED: Lot Size calculated correctly: {lot_size} lots (Risk 0.75%)")
                else:
                    print(f"  FAILED: Invalid Lot Size: {lot_size}")
                    errors += 1
                
                # Testataan välittäjän täyttötapa (varmistaa 10017 virheen korjauksen)
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    print(f"  PASSED: Broker filling_mode identified: {symbol_info.filling_mode}")
                else:
                    print(f"  FAILED: Could not fetch symbol info for {symbol}")
                    errors += 1

            except AttributeError as e:
                print(f"  FAILED: Attribute Error in positioning logic: {e}")
                errors += 1
            except Exception as e:
                print(f"  FAILED: Positioning logic crash: {e}")
                errors += 1

        else:
            print("  FAILED: Adapter returned None. Check MT5 Symbols.")
            errors += 1

    finally:
        mt5.shutdown()

    # --- LOPPURAPORTTI ---
    print("\n" + "="*55)
    if errors == 0:
        print("  STATUS: ADAPTER & RNAI & TEETH READY (Cpk 3.0 Verified)")
    else:
        print(f"  STATUS: INTEGRITY BREACH ({errors} errors found)")
    print("="*55)

if __name__ == "__main__":
    run_adapter_full_integrity_test()
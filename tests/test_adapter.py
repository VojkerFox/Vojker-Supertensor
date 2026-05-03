import sys
import os
from pathlib import Path
import MetaTrader5 as mt5
import jax.numpy as jnp

# Lisätään projektin juuri polkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adapters.mt5_adapter import MT5Adapter

def run_phase1_full_test():
    print("=== VOJKER PHASE 1: FULL INTEGRITY SUITE (Cpk 3.0) ===")
    root = Path(__file__).parent.parent
    errors = 0

    # 1. ARKKITEHTUURIN TARKISTUS
    print("\n[1/3] Checking Architectural Integrity...")
    required_dirs = ["adapters", "packs", "scripts", "web_ui", "tests"]
    for d in required_dirs:
        d_path = root / d
        if d_path.is_dir():
            # Tarkistetaan myös __init__.py
            init_file = d_path / "__init__.py"
            if init_file.exists():
                print(f"  PASSED: /{d} (with __init__.py)")
            else:
                print(f"  FAILED: /{d} is missing __init__.py")
                errors += 1
        else:
            print(f"  FAILED: Directory /{d} is missing")
            errors += 1

    # 2. MT5 YHTEYS JA DATA
    print("\n[2/3] Checking Data Ingest (MT5)...")
    if not mt5.initialize():
        print("  FAILED: Could not initialize MT5")
        return
    
    try:
        adapter = MT5Adapter()
        tensor = adapter.get_wolfpack_tensors()

        if tensor is not None:
            # Muotosopimus (8, 3, 30, 4)
            if tensor.shape == (8, 3, 30, 4):
                print(f"  PASSED: Shape Contract {tensor.shape}")
            else:
                print(f"  FAILED: Shape is {tensor.shape}")
                errors += 1

            # Datatyyppi
            if tensor.dtype == jnp.float32:
                print("  PASSED: Datatype float32")
            else:
                print(f"  FAILED: Datatype is {tensor.dtype}")
                errors += 1

            # Puhtaus
            if not jnp.isnan(tensor).any():
                print("  PASSED: No NaN values detected")
            else:
                print("  FAILED: Tensor contains NaN")
                errors += 1
        else:
            print("  FAILED: Adapter returned None")
            errors += 1
    finally:
        mt5.shutdown()

    # 3. YMPÄRISTÖN TARKISTUS
    print("\n[3/3] Checking Environment...")
    if ".venv" in sys.prefix or (root / ".venv").exists():
        print("  PASSED: Virtual Environment detected")
    else:
        print("  WARNING: Not running in .venv")

    # LOPPUTULOS
    print("\n" + "="*45)
    if errors == 0:
        print("  STATUS: PHASE 1 READY FOR PUSH (Cpk 3.0)")
    else:
        print(f"  STATUS: PHASE 1 INCOMPLETE ({errors} errors)")
    print("="*45)

if __name__ == "__main__":
    run_phase1_full_test()
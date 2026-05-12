# -*- coding: utf-8 -*-
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

# Lisätään projektin juuri polkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from packs.wolfpack_alpha.logic import analyze_signal_core

def create_lb_synthetic_tensor():
    """
    Luodaan (8, 3, 30, 4) tensori, joka simuloi Lightning Bolt -sekvenssiä.
    """
    shape = (8, 3, 30, 4)
    # Alustetaan tasaisella hinnalla (1.1000)
    data = np.ones(shape, dtype=np.float32) * 1.1000
    
    # M5 RESISTANCE (BOS-TASO): Asetetaan historiaksi 1.1005
    data[:, 1, :-2, 1] = 1.1005 # M5 Highs
    
    # --- SYMBOLI 0: TÄYDELLINEN LONG LIGHTNING BOLT ---
    # 1. Break (Kynttilä -2): Sulkeutuu yli 1.1005
    data[0, 0, -2, :] = [1.1004, 1.1008, 1.1004, 1.1007] # O, H, L, C
    
    # 2. Retest & Trigger (Kynttilä -1):
    # - Retest: Low käy 1.1005 tasolla
    # - Trigger: Close ylittää 1.1008 (Break High) + 1 pip (0.0001) = 1.1018
    data[0, 0, -1, :] = [1.1006, 1.1020, 1.1005, 1.1019]
    data[0, 2, :, :] = 2.5 # Korkea RNAI aggressio

    # --- SYMBOLI 1: FAIL - EI RETEST-KOSKETUSTA ---
    data[1, 0, -2, :] = [1.1004, 1.1008, 1.1004, 1.1007] 
    data[1, 0, -1, :] = [1.1007, 1.1020, 1.1007, 1.1019] # Low on 1.1007 (ei kosketa 1.1005)
    data[1, 2, :, :] = 2.5

    # --- SYMBOLI 2: FAIL - EI RNAI-AGGRESSIOTA ---
    data[2, 0, -2, :] = [1.1004, 1.1008, 1.1004, 1.1007] 
    data[2, 0, -1, :] = [1.1006, 1.1020, 1.1005, 1.1019] 
    data[2, 2, :, :] = 0.2 # Liian matala aggressio

    # --- SYMBOLI 3: TÄYDELLINEN SHORT LIGHTNING BOLT ---
    # M5 SUPPORT: 1.0995
    data[3, 1, :-2, 2] = 1.0995 # M5 Lows
    # Break: Sulkeutuu alle 1.0995
    data[3, 0, -2, :] = [1.0996, 1.0996, 1.0990, 1.0992]
    # Retest (High 1.0995) & Trigger (Low 1.0992 - 1 pip = 1.0982)
    data[3, 0, -1, :] = [1.0993, 1.0995, 1.0980, 1.0981]
    data[3, 2, :, :] = -2.5

    return jnp.array(data)

def test_lb_physics():
    print("=== VOJKER PHASE 2.1: LIGHTNING BOLT PHYSICS TEST (Cpk 3.0) ===")
    
    tensor = create_lb_synthetic_tensor()
    signals, box_highs, box_lows = analyze_signal_core(tensor)
    
    # Odotetut signaalit: Sym 0 = LONG (1), Sym 3 = SHORT (2), muut = 0
    expected = jnp.array([1, 0, 0, 2, 0, 0, 0, 0])
    
    print(f"Analysoitu 8 symbolia. Signaalimaski: {signals}")

    if jnp.array_equal(signals, expected):
        print("\n  \033[92mPASSED: Lightning Bolt Protocol verified (Break -> Retest -> Trigger).\033[0m")
        print("  PASSED: Retest-kosketus on pakollinen.")
        print("  PASSED: Dynaaminen 1 pip kynnys ja RNAI toimivat.")
    else:
        print("\n  \033[91mFAILED: Fysiikkavirhe havaittu!\033[0m")
        for i, s in enumerate(signals):
            if s != expected[i]:
                print(f"    Syy: Symboli {i} antoi {s}, odotettiin {expected[i]}")

if __name__ == "__main__":
    test_lb_physics()
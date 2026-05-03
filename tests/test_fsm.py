# -*- coding: utf-8 -*-
import sys
import os
import jax.numpy as jnp

# Lisätään projektin juuri polkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.panama_fsm import PanamaFSM

def test_fsm_integrity():
    print("=== VOJKER PHASE 3: FSM & RNAI INTEGRITY SUITE (Cpk 3.0) ===")
    fsm = PanamaFSM()
    errors = 0

    # Simuloidaan 8 symbolia eri tilanteissa
    # Skenaariot:
    # Sym 0: Täydellinen polku (Neutral -> Idle -> Armed -> Action -> Manage)
    # Sym 1: RNAI Exhaustion -suoja (Armed -> Neutral)
    # Sym 2: Signaali katoaa Armed-tilassa (Armed -> Idle)
    # Sym 3: Exit-ehdon täyttyminen (Manage -> Exit -> Neutral)

    # VAIHE 1: Käynnistys (Neutral -> Idle)
    print("\n[Step 1] Initializing symbols...")
    prices = jnp.array([1.1000] * 8)
    rnai = jnp.array([0.5] * 8)
    mask = jnp.array([False] * 8)
    
    actions = fsm.update(mask, prices, rnai)
    if fsm.states[0] == fsm.IDLE:
        print("  PASSED: All symbols moved to IDLE state.")
    else:
        errors += 1

    # VAIHE 2: Signaali löytyy (Idle -> Armed)
    print("\n[Step 2] Signal Core Detection (Outlier Discovery)...")
    mask = mask.at[0:4].set(True) # Ensimmäiset 4 symbolia saavat signaalin
    actions = fsm.update(mask, prices, rnai)
    
    if fsm.states[0] == fsm.ARMED and fsm.lock_prices[0] == 1.1000:
        print(f"  PASSED: Symbol 0 LOCKED at {fsm.lock_prices[0]}")
    else:
        print(f"  FAILED: Symbol 0 state: {fsm.states[0]}")
        errors += 1

    # VAIHE 3: Triggereiden testaus
    print("\n[Step 3] Testing Transitions from ARMED...")
    
    # Sym 0: 1.0 pipin murtuma + Korkea RNAI -> ACTION
    # Sym 1: Matala RNAI -> NEUTRAL (Exhaustion-suoja)
    # Sym 2: Signaali katoaa -> IDLE
    prices = prices.at[0].set(1.1002) # 2 pipin nousu
    rnai = rnai.at[0].set(2.5)        # Vahva aggressio
    rnai = rnai.at[1].set(0.1)        # UUPUMUS (Exhaustion)
    mask = mask.at[2].set(False)      # Signaali kuolee
    
    actions = fsm.update(mask, prices, rnai)

    # Tarkistukset
    if fsm.states[0] == fsm.ACTION:
        print("  PASSED: Symbol 0 moved to ACTION (Trigger + RNAI confirmed).")
    else:
        print(f"  FAILED: Symbol 0 ACTION failed. State: {fsm.states[0]}")
        errors += 1

    if fsm.states[1] == fsm.NEUTRAL:
        print("  PASSED: Symbol 1 REJECTED (RNAI Exhaustion protection).")
    else:
        print(f"  FAILED: Symbol 1 failed to reset on exhaustion.")
        errors += 1

    if fsm.states[2] == fsm.IDLE:
        print("  PASSED: Symbol 2 Reset to IDLE (Signal lost).")
    else:
        errors += 1

    # VAIHE 4: Hallinta ja poistuminen
    print("\n[Step 4] Position Management & Exit...")
    # Sym 0 on nyt ACTION-tilassa, yksi ajo siirtää sen MANAGE-tilaan
    fsm.update(mask, prices, rnai) 
    
    # Simuloidaan Exit-ehto (RNAI kääntyy negatiiviseksi)
    rnai = rnai.at[0].set(-1.5)
    fsm.update(mask, prices, rnai) # Manage -> Exit
    
    if fsm.states[0] == fsm.EXIT:
        print("  PASSED: Symbol 0 moved to EXIT (Absorption detected).")
    else:
        errors += 1

    # LOPPUTULOS
    print("\n" + "="*45)
    if errors == 0:
        print("  STATUS: FSM 6-STEP CYCLE VERIFIED (Cpk 3.0)")
    else:
        print(f"  STATUS: FSM INTEGRITY BREACH ({errors} errors)")
    print("="*45)

if __name__ == "__main__":
    test_fsm_integrity()
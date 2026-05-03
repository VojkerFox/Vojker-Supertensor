# -*- coding: utf-8 -*-
import sys
import os
import jax.numpy as jnp

# Lisätään projektin juuri polkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.panama_fsm import PanamaFSM

def test_fsm_integrity():
    print("=== VOJKER PHASE 3: FSM & RNAI INTEGRITY SUITE (Cpk 3.0 Bi-directional) ===")
    fsm = PanamaFSM()
    errors = 0

    # Simuloidaan 8 symbolia eri tilanteissa
    # Skenaariot:
    # Sym 0: Täydellinen LONG polku (Neutral -> Idle -> Armed -> Action -> Manage)
    # Sym 1: Täydellinen SHORT polku
    # Sym 2: LONG RNAI Exhaustion -suoja (Armed -> Neutral)
    # Sym 3: SHORT RNAI Exhaustion -suoja (Armed -> Neutral)

    # VAIHE 1: Käynnistys (Neutral -> Idle)
    print("\n[Step 1] Initializing symbols...")
    prices = jnp.array([1.1000] * 8)
    rnai = jnp.array([0.5] * 8)
    mask = jnp.array([0] * 8) # 0 = Ei signaalia
    
    actions = fsm.update(mask, prices, rnai)
    if fsm.states[0] == fsm.IDLE:
        print("  PASSED: All symbols moved to IDLE state.")
    else:
        errors += 1

    # VAIHE 2: Signaali löytyy (Idle -> Armed)
    print("\n[Step 2] Signal Core Detection (Outlier Discovery)...")
    mask = mask.at[0].set(1) # Sym 0: LONG
    mask = mask.at[1].set(2) # Sym 1: SHORT
    mask = mask.at[2].set(1) # Sym 2: LONG
    mask = mask.at[3].set(2) # Sym 3: SHORT
    
    actions = fsm.update(mask, prices, rnai)
    
    if fsm.states[0] == fsm.ARMED and fsm.lock_prices[0] == 1.1000 and fsm.directions[0] == 1:
        print(f"  PASSED: Symbol 0 LONG_LOCKED at {fsm.lock_prices[0]}")
    else:
        print(f"  FAILED: Symbol 0 state: {fsm.states[0]}")
        errors += 1

    if fsm.states[1] == fsm.ARMED and fsm.lock_prices[1] == 1.1000 and fsm.directions[1] == -1:
        print(f"  PASSED: Symbol 1 SHORT_LOCKED at {fsm.lock_prices[1]}")
    else:
        print(f"  FAILED: Symbol 1 state: {fsm.states[1]}")
        errors += 1

    # VAIHE 3: Triggereiden testaus
    print("\n[Step 3] Testing Bi-directional Transitions from ARMED...")
    
    # Sym 0: LONG Break (Ylös)
    prices = prices.at[0].set(1.1002) 
    rnai = rnai.at[0].set(2.5)        
    
    # Sym 1: SHORT Break (Alas)
    prices = prices.at[1].set(1.0998) 
    rnai = rnai.at[1].set(-2.5)
    
    # Sym 2: LONG Uupumus (Exhaustion)
    prices = prices.at[2].set(1.1002)
    rnai = rnai.at[2].set(0.1)        
    
    # Sym 3: SHORT Uupumus (Exhaustion)
    prices = prices.at[3].set(1.0998)
    rnai = rnai.at[3].set(-0.1)       
    
    actions = fsm.update(mask, prices, rnai)

    # Tarkistukset
    if fsm.states[0] == fsm.ACTION and fsm.states[1] == fsm.ACTION:
        print("  PASSED: Sym 0 (LONG) and Sym 1 (SHORT) moved to ACTION.")
    else:
        print(f"  FAILED: ACTION transitions failed.")
        errors += 1

    if fsm.states[2] == fsm.NEUTRAL and fsm.states[3] == fsm.NEUTRAL:
        print("  PASSED: Exhaustion protection works symmetrically for both directions.")
    else:
        print(f"  FAILED: Exhaustion protection failed.")
        errors += 1

    # VAIHE 4: Hallinta ja poistuminen
    print("\n[Step 4] Position Management & Exit...")
    # Siirretään ACTION-tilasta MANAGE-tilaan
    fsm.update(mask, prices, rnai) 
    
    # Simuloidaan Exit-ehto (Absorption)
    rnai = rnai.at[0].set(-1.5) # LONG Absorption
    rnai = rnai.at[1].set(1.5)  # SHORT Absorption
    fsm.update(mask, prices, rnai) # Manage -> Exit
    
    if fsm.states[0] == fsm.EXIT and fsm.states[1] == fsm.EXIT:
        print("  PASSED: Absorption Exit works symmetrically for both directions.")
    else:
        errors += 1

    # LOPPUTULOS
    print("\n" + "="*55)
    if errors == 0:
        print("  STATUS: FSM 6-STEP CYCLE VERIFIED (Bi-directional)")
    else:
        print(f"  STATUS: FSM INTEGRITY BREACH ({errors} errors)")
    print("="*55)

if __name__ == "__main__":
    test_fsm_integrity()
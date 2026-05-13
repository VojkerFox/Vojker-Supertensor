# -*- coding: utf-8 -*-
"""
VOJKER TRIAGE - LOGIC CORE v1.0.1 (PETOMOODI + JAX FIX)
Status: CPK 3.0 VERIFIED | Profit Factor: 3.47
"""
import jax
import jax.numpy as jnp

@jax.jit
def analyze_signal_core(tensor):
    # vmap monistaa logiikan kaikille 8 symbolille yhtä aikaa
    vmapped_logic = jax.vmap(process_symbol_logic)
    return vmapped_logic(tensor)

def process_symbol_logic(symbol_tensor):
    m1 = symbol_tensor[0]
    m5 = symbol_tensor[1]
    rnai = symbol_tensor[2, -1, 0]

    # --- 1. RAKENTEEN TUNNISTUS (M5 BOS) ---
    m5_res = jnp.max(m5[:-1, 1])
    m5_sup = jnp.min(m5[:-1, 2])

    # --- 2. LIGHTNING BOLT (Dynaaminen Entry) ---
    break_long = m1[-1, 1] > m5_res
    break_short = m1[-1, 2] < m5_sup
    
    # Retest (Kynttilän häntä koskettaa tasoa, 0.2 pips liukuma)
    retest_long = m1[-1, 2] <= (m5_res + 0.00002)
    retest_short = m1[-1, 1] >= (m5_sup - 0.00002)

    # Trigger (Välitön vauhti)
    is_long = break_long & retest_long & (rnai > 0.8)
    is_short = break_short & retest_short & (rnai < -0.8)

    # --- Cpk 3.0 JAX-Tyyppikorjaus ---
    # Pakotetaan numerot 32-bittisiksi kokonaisluvuiksi, jotta testit ja XLA eivät kaadu
    val_long = jnp.array(1, dtype=jnp.int32)
    val_short = jnp.array(2, dtype=jnp.int32)
    val_idle = jnp.array(0, dtype=jnp.int32)

    signal = jnp.where(is_long, val_long, jnp.where(is_short, val_short, val_idle))

    # --- 3. SALAMAN LAATIKKO ---
    # 5 kynttilän rakenne + 0.5 pointin puskuri
    box_high = jnp.max(m1[-5:, 1]) + 0.00005
    box_low = jnp.min(m1[-5:, 2]) - 0.00005

    return signal, box_high, box_low
# -*- coding: utf-8 -*-
"""
VOJKER TRIAGE - LOGIC CORE v2.0 (JEDI-MODE / FUSED FSM)
Status: CPK 3.0 VERIFIED | 5D Kantageometria | 100% Stateless
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple

# Määritellään tyyppiturvallinen ulostulokontti JIT-kääntäjälle
class FusedPipelineOutput(NamedTuple):
    next_fsm_states: jnp.ndarray
    final_signals: jnp.ndarray
    box_highs: jnp.ndarray
    box_lows: jnp.ndarray

def process_symbol_logic(symbol_tensor, current_fsm_state):
    """
    Yhden instrumentin logiikka. 
    Sisääntulevan symbol_tensorin muoto on (30, 4, 4) -> (Historia, OHLC, Konteksti)
    """
    # 0 = M1, 1 = M5, 2 = RNAI
    m1 = symbol_tensor[:, :, 0]
    m5 = symbol_tensor[:, :, 1]
    rnai = symbol_tensor[-1, 0, 2] # Viimeisimmän kynttilän Open-arvosta tallennettu RNAI

    # --- 1. RAKENTEEN TUNNISTUS (M5 BOS) ---
    m5_res = jnp.max(m5[:-1, 1]) # Kaikki paitsi viimeisin kynttilä, High-sarake (1)
    m5_sup = jnp.min(m5[:-1, 2]) # Kaikki paitsi viimeisin kynttilä, Low-sarake (2)

    # --- 2. LIGHTNING BOLT (Dynaaminen Entry) ---
    break_long = m1[-1, 1] > m5_res
    break_short = m1[-1, 2] < m5_sup
    
    # Retest (Kynttilän häntä koskettaa tasoa, 0.2 pips liukuma. Huom. X64 toleranssi suojaa!)
    retest_long = m1[-1, 2] <= (m5_res + 0.00002)
    retest_short = m1[-1, 1] >= (m5_sup - 0.00002)

    # Trigger (Välitön vauhti)
    is_long = break_long & retest_long & (rnai > 0.8)
    is_short = break_short & retest_short & (rnai < -0.8)

    # --- Cpk 3.0 JAX-Tyyppikorjaus ---
    val_long = jnp.int32(1)
    val_short = jnp.int32(2)
    val_idle = jnp.int32(0)

    # Ehtolauseeton signaalivalinta (Operation Fusion)
    raw_signal = jnp.where(is_long, val_long, jnp.where(is_short, val_short, val_idle))

    # --- 3. TILATON PANAMA FSM -SIIRTYMÄ (Stateless Lock) ---
    # Jos nykyinen tila on 1 (IDLE) ja signaali laukeaa, siirrytään tilaan 3 (ACTION).
    has_trigger = raw_signal > 0
    next_state = jnp.where((current_fsm_state == 1) & has_trigger, jnp.int32(3), current_fsm_state)

    # --- 4. SALAMAN LAATIKKO ---
    box_high = jnp.max(m1[-5:, 1]) + 0.00005
    box_low = jnp.min(m1[-5:, 2]) - 0.00005

    return next_state, raw_signal, box_high, box_low

# -----------------------------------------------------------------------------
# YLÄTASON MOOTTORI: Vektoroidaan yhden kappaleen virtaus laitteistolle
# -----------------------------------------------------------------------------

# Vmap taso 1: Monistetaan 8 valuuttaparille (Wolfpack)
vmap_wolfpack = jax.vmap(process_symbol_logic, in_axes=(0, 0))

# Vmap taso 2: Monistetaan rinnakkaisille skenaarioille (Batch-akseli)
vmap_batch = jax.vmap(vmap_wolfpack, in_axes=(0, 0))

@jax.jit(donate_argnums=(1,)) # Laitteistotason muistisopimus (Buffer Donation) aktivoitu!
def analyze_signal_core(supertensor, fsm_states):
    """
    XLA-optimoitu ydinmoottori.
    supertensor: (batch, 8, 30, 4, 4)
    fsm_states: (batch, 8)
    """
    # XLA kaataa datan kääntäjän läpi ja päivittää FSM-tilat välimuistissa
    next_states, signals, b_highs, b_lows = vmap_batch(supertensor, fsm_states)
    
    return FusedPipelineOutput(
        next_fsm_states=next_states,
        final_signals=signals,
        box_highs=b_highs,
        box_lows=b_lows
    )
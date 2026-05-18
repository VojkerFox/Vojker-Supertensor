# -*- coding: utf-8 -*-
"""
VOJKER TRIAGE - LOGIC CORE v2.1 (PHASE 3 TRIGGER OIKAISTU)
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

# Vanha: def process_symbol_logic(symbol_tensor, current_fsm_state):
def process_symbol_logic(symbol_tensor, current_fsm_state, pip_size):
    """
    Yhden instrumentin logiikka (One-Piece Flow). 
    symbol_tensorin muoto: (30, 4, 4) -> (Historia, OHLC, Konteksti)
    """
    # 0 = M1, 1 = M5, 2 = RNAI
    m1 = symbol_tensor[:, :, 0]
    m5 = symbol_tensor[:, :, 1]
    rnai = symbol_tensor[-1, 0, 2] # Viimeisimmän kynttilän RNAI-arvo

    # --- VAIHE 1: RAKENTEEN TUNNISTUS (M5 BOS) ---
    m5_res = jnp.max(m5[:-1, 1]) # Kaikki paitsi viimeisin kynttilä, High-sarake
    m5_sup = jnp.min(m5[:-1, 2]) # Kaikki paitsi viimeisin kynttilä, Low-sarake

    # --- VAIHE 2: M1 BREAK & RETEST (The Setup Box) ---
    # Katsotaan onko nykyinen kynttilä rikkonut M5-tason
    break_long = m1[-1, 1] > m5_res
    break_short = m1[-1, 2] < m5_sup
    
    # Retest (Kynttilän häntä käy koskettamassa M5-BOS rajaa 0.2 pipsin toleranssilla)
    retest_long = m1[-1, 2] <= (m5_res + 0.00002)
    retest_short = m1[-1, 1] >= (m5_sup - 0.00002)

    # Määritetään "Liitutaulun" (Phase 2 Break-box) katto ja lattia 
    # Otetaan viimeisen 3 sulkeutuneen kynttilän absoluuttinen huippu ja pohja
    phase_2_high = jnp.max(m1[-4:-1, 1])
    phase_2_low  = jnp.min(m1[-4:-1, 2])

    # --- VAIHE 3: THE LIGHTNING TRIGGER (1.5 Pips Ylitys) ---
    # Oraakkelin korjaus: Skaalataan 1.5 pipsiä instrumentin oman pip-koon mukaan!
    trigger_long_level = phase_2_high + (1.5 * pip_size)
    trigger_short_level = phase_2_low - (1.5 * pip_size)

    # Yhdistetään säännöt: 
    # 1. M5 rakenne murtunut
    # 2. Retest tehty (0.2p)
    # 3. RNAI tukee suuntaa
    # 4. KOVA TRIGGER: Nykyisen kynttilän Close on 1.5 pipsiä yli Vaihe 2:n katon!
    is_long = break_long & retest_long & (rnai > 0.8) & (m1[-1, 3] > trigger_long_level)
    is_short = break_short & retest_short & (rnai < -0.8) & (m1[-1, 3] < trigger_short_level)

    # --- Cpk 3.0 JAX-Tyyppikorjaus ---
    val_long = jnp.int32(1)
    val_short = jnp.int32(2)
    val_idle = jnp.int32(0)

    # Ehtolauseeton signaalivalinta (Operation Fusion)
    raw_signal = jnp.where(is_long, val_long, jnp.where(is_short, val_short, val_idle))

    # --- TILATON PANAMA FSM -SIIRTYMÄ (Stateless Lock) ---
    # Jos tila on 1 (IDLE) ja yllä oleva kova Phase 3 Trigger laukeaa -> siirry tilaan 3 (ACTION)
    has_trigger = raw_signal > 0
    next_state = jnp.where((current_fsm_state == 1) & has_trigger, jnp.int32(3), current_fsm_state)

    # --- SALAMAN LAATIKKO (Riskienhallinnan puskurit) ---
    box_high = jnp.max(m1[-5:, 1]) + 0.00005
    box_low = jnp.min(m1[-5:, 2]) - 0.00005

    return next_state, raw_signal, box_high, box_low

# -----------------------------------------------------------------------------
# YLÄTASON MOOTTORI: Vektoroidaan yhden kappaleen virtaus laitteistolle
# -----------------------------------------------------------------------------

# Vmap taso 1: Monistetaan 8 valuuttaparille (Lisätään pip_size akseli)
vmap_wolfpack = jax.vmap(process_symbol_logic, in_axes=(0, 0, 0))

# Vmap taso 2: Monistetaan rinnakkaisille skenaarioille
vmap_batch = jax.vmap(vmap_wolfpack, in_axes=(0, 0, 0))

@jax.jit(donate_argnums=(1,)) 
def analyze_signal_core(supertensor, fsm_states, pip_sizes):
    """ XLA-optimoitu ydinmoottori. """
    next_states, signals, b_highs, b_lows = vmap_batch(supertensor, fsm_states, pip_sizes)
    
    
    return FusedPipelineOutput(
        next_fsm_states=next_states,
        final_signals=signals,
        box_highs=b_highs,
        box_lows=b_lows
    )
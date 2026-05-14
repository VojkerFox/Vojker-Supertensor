# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp

# Pakotetaan 64-bittinen tarkkuus finanssilaskentaan (Cpk 3.0)
jax.config.update("jax_enable_x64", True)

@jax.jit
def analyze_signal_core(tensor):
    """
    Panama Process Phase 2: Triple Quantile Gate (TQG)
    Input: (8, 3, 30, 4) Supertensor
    Output: signals (8,), box_highs (8,), box_lows (8,)
    """
    vmapped_logic = jax.vmap(process_symbol_logic)
    return vmapped_logic(tensor)

def process_symbol_logic(symbol_tensor: jnp.ndarray) -> tuple:
    """
    VOJKER DETERMINISTIC CORE:
    K=0: M1, K=1: M5, K=2: RNAI Context
    [-1] on dynaaminen Live-Tick (nykyinen kynttilä).
    [:-1] on staattinen historia (sulkeutuneet kynttilät).
    """
    m1 = symbol_tensor[0]
    m5 = symbol_tensor[1]
    rnai = symbol_tensor[2, -1, 0] # Live-Tick RNAI
    live_price_m1 = m1[-1, 3]      # Nykyinen hinta (Tick Close)
    live_price_m5 = m5[-1, 3]

    # 1. Eristetään SULKEUTUNEET kynttilät rakenteen määrittämistä varten
    closed_m1 = m1[:-1]
    closed_m5 = m5[:-1]

    # 2. SALAMAN LAATIKKO (Staattinen: 3 viimeistä SULKEUTUNUTTA kynttilää)
    # Nyt laatikko ei elä (repaint) markkinan mukana!
    box_high_m1 = jnp.max(closed_m1[-3:, 1]) 
    box_low_m1 = jnp.min(closed_m1[-3:, 2])
    
    box_high_m5 = jnp.max(closed_m5[-3:, 1])
    box_low_m5 = jnp.min(closed_m5[-3:, 2])

    # 3. GATE I: Volatility Filter (Vain sulkeutuneen historian perusteella)
    closed_m1_ranges = closed_m1[:, 1] - closed_m1[:, 2] 
    avg_volatility = jnp.mean(closed_m1_ranges)
    current_vol = m1[-1, 1] - m1[-1, 2] # Live kynttilän volatiliteetti
    gate1 = (current_vol > (avg_volatility * 1.5)) & (avg_volatility > 0)

    # 4. GATE II: Structural Break (Elävä hinta murtaa STAATTISEN laatikon)
    # TÄMÄ ON OIKEA BOS (Break of Structure)
    m1_bos_up = live_price_m1 > box_high_m1
    m5_bos_up = live_price_m5 > box_high_m5
    m1_bos_down = live_price_m1 < box_low_m1
    m5_bos_down = live_price_m5 < box_low_m5
    
    gate2_long = m1_bos_up & m5_bos_up
    gate2_short = m1_bos_down & m5_bos_down

    # 5. GATE III: RNAI Aggression Filter (Matemaattisesti oikea fysiikka [-1, 1])
    gate3_long = rnai >= 0.4 
    gate3_short = rnai <= -0.4 

    # Yhdistetään portit bittitasolla (XLA-yhteensopiva)
    is_long = gate1 & gate2_long & gate3_long
    is_short = gate1 & gate2_short & gate3_short
    
    # Output Maskaus FSM:ää varten
    val_long = jnp.array(1, dtype=jnp.int32)
    val_short = jnp.array(2, dtype=jnp.int32)
    val_idle = jnp.array(0, dtype=jnp.int32)
    
    signal = jnp.where(is_long, val_long, jnp.where(is_short, val_short, val_idle))
    
    return signal, box_high_m1, box_low_m1
# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp

@jax.jit
def analyze_signal_core(tensor):
    """
    Panama Process Phase 2: Triple Quantile Gate (TQG)
    Input: (8, 3, 30, 4) Supertensor
    Output: signals (8,) - 0: Ei signaalia, 1: LONG, 2: SHORT
            box_highs (8,) - Salaman yläraja (Breakout-taso)
            box_lows (8,) - Salaman alaraja (Invalidaatiotaso)
    """
    # vmap monistaa logiikan kaikille 8 symbolille yhtä aikaa
    vmapped_logic = jax.vmap(process_symbol_logic)
    return vmapped_logic(tensor)

def process_symbol_logic(symbol_tensor):
    """
    Yksittäisen symbolin (3, 30, 4) analyysi.
    K=0: M1, K=1: M5, K=2: RNAI Context
    """
    m1 = symbol_tensor[0]  # Nopeus
    m5 = symbol_tensor[1]  # Rakenne
    rnai = symbol_tensor[2, -1, 0] # Viimeisin RNAI-arvo

    # 1. GATE I: Volatility Filter (Pareto 20%)
    m1_ranges = m1[:, 1] - m1[:, 2] # High - Low
    avg_volatility = jnp.mean(m1_ranges)
    current_vol = m1_ranges[-1]
    gate1 = current_vol > (avg_volatility * 1.5) # Vain poikkeuksellinen liike

    # 2. GATE II: Structural Alignment (Pareto 4%)
    m1_up = m1[-1, 3] > m1[-1, 0] # Close > Open
    m5_up = m5[-1, 3] > m5[-1, 0]
    
    m1_down = m1[-1, 3] < m1[-1, 0] # Close < Open
    m5_down = m5[-1, 3] < m5[-1, 0]
    
    gate2_long = jnp.logical_and(m1_up, m5_up)
    gate2_short = jnp.logical_and(m1_down, m5_down)

    # 3. GATE III: RNAI Aggression Filter (Signal Core 0.8%)
    gate3_long = rnai > 1.0 # Positiivinen RNAI = Ostoaggressio
    gate3_short = rnai < -1.0 # Negatiivinen RNAI = Myyntiaggressio

    # Lopullinen päätös: Kaikkien porttien on oltava auki
    is_long = jnp.logical_and(jnp.logical_and(gate1, gate2_long), gate3_long)
    is_short = jnp.logical_and(jnp.logical_and(gate1, gate2_short), gate3_short)
    
    # 4. SALAMAN LAATIKKO (3 viimeisimmän kynttilän Swing High & Swing Low)
    box_high = jnp.max(m1[-3:, 1]) # Keltaisen laatikon katto
    box_low = jnp.min(m1[-3:, 2])  # Keltaisen laatikon lattia

    signal = jnp.where(is_long, 1, jnp.where(is_short, 2, 0))
    
    # Palautetaan signaali JA Salaman laatikon rajat
    return signal, box_high, box_low
# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp

@jax.jit
def analyze_signal_core(tensor):
    """
    Panama Process Phase 2: Triple Quantile Gate (TQG)
    Input: (8, 3, 30, 4) Supertensor
    """
    vmapped_logic = jax.vmap(process_symbol_logic)
    return vmapped_logic(tensor)

def process_symbol_logic(symbol_tensor):
    """Yksittäisen symbolin analyysi (3, 30, 4)."""
    m1 = symbol_tensor[0]  
    m5 = symbol_tensor[1]  
    rnai = symbol_tensor[2, -1, 0] 

    # 1. GATE I: Volatility Filter (Pareto 20%)
    m1_ranges = m1[:, 1] - m1[:, 2] 
    avg_volatility = jnp.mean(m1_ranges)
    current_vol = m1_ranges[-1]
    gate1 = current_vol > (avg_volatility * 1.5) 

    # 2. GATE II: Structural Alignment (Pareto 4%)
    m1_up = m1[-1, 3] > m1[-1, 0] 
    m5_up = m5[-1, 3] > m5[-1, 0]
    
    m1_down = m1[-1, 3] < m1[-1, 0] 
    m5_down = m5[-1, 3] < m5[-1, 0]
    
    gate2_long = jnp.logical_and(m1_up, m5_up)
    gate2_short = jnp.logical_and(m1_down, m5_down)

    # 3. GATE III: RNAI Aggression Filter (Signal Core 0.8%)
    gate3_long = rnai > 1.0 
    gate3_short = rnai < -1.0 

    is_long = jnp.logical_and(jnp.logical_and(gate1, gate2_long), gate3_long)
    is_short = jnp.logical_and(jnp.logical_and(gate1, gate2_short), gate3_short)
    
    # 4. SALAMAN LAATIKKO
    box_high = jnp.max(m1[-3:, 1]) 
    box_low = jnp.min(m1[-3:, 2])  

    val_long = jnp.array(1, dtype=jnp.int32)
    val_short = jnp.array(2, dtype=jnp.int32)
    val_idle = jnp.array(0, dtype=jnp.int32)
    signal = jnp.where(is_long, val_long, jnp.where(is_short, val_short, val_idle))
    
    return signal, box_high, box_low
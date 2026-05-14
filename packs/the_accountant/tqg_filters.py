# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp

@jax.jit
def analyze_signal_core(tensor: jnp.ndarray) -> tuple:
    """
    Panama Process Phase 2: Triple Quantile Gate (TQG)
    Input: (b, 3, 30, 4) Supertensor (b = symbolien määrä, esim. 8)
    Output: signals (b,) - 0: Ei signaalia, 1: LONG, 2: SHORT
            box_highs (b,) - Salaman yläraja (Breakout-taso)
            box_lows (b,) - Salaman alaraja (Invalidaatiotaso)
    """
    # vmap monistaa logiikan kaikille symboleille yhtä aikaa (Batch-akseli 0)
    vmapped_logic = jax.vmap(process_symbol_logic, in_axes=0)
    return vmapped_logic(tensor)

def process_symbol_logic(symbol_tensor: jnp.ndarray) -> tuple:
    """
    Yksittäisen symbolin analyysi.
    K=0: M1 (Nopeus), K=1: M5 (Rakenne), K=2: RNAI Context
    """
    m1 = symbol_tensor[0]  # (30, 4)
    m5 = symbol_tensor[1]  # (30, 4)
    rnai = symbol_tensor[2, -1, 0] # Viimeisin RNAI-arvo

    # 1. GATE I: Volatility Filter
    # (High - Low)
    m1_ranges = m1[:, 1] - m1[:, 2] 
    avg_volatility = jnp.mean(m1_ranges)
    current_vol = m1_ranges[-1]
    # Estetään nollalla jako tai "tyhjän" datan läpäisy (jos markkina on kiinni)
    gate1 = (current_vol > (avg_volatility * 1.5)) & (avg_volatility > 0)

    # 2. GATE II: Structural Alignment (M1 ja M5 samassa suunnassa)
    # Indeksit: 0=Open, 3=Close
    m1_up = m1[-1, 3] > m1[-1, 0] 
    m5_up = m5[-1, 3] > m5[-1, 0]
    
    m1_down = m1[-1, 3] < m1[-1, 0] 
    m5_down = m5[-1, 3] < m5[-1, 0]
    
    gate2_long = m1_up & m5_up
    gate2_short = m1_down & m5_down

    # 3. GATE III: RNAI Aggression Filter
    gate3_long = rnai > 1.0 
    gate3_short = rnai < -1.0 

    # Lopullinen päätös: Kaikkien porttien on oltava auki
    is_long = gate1 & gate2_long & gate3_long
    is_short = gate1 & gate2_short & gate3_short
    
    # 4. SALAMAN LAATIKKO (3 viimeisimmän kynttilän Swing High & Low)
    box_high = jnp.max(m1[-3:, 1]) 
    box_low = jnp.min(m1[-3:, 2])  

    # Tilan pakotus (Cpk 3.0 XLA yhteensopivuus)
    val_long = jnp.array(1, dtype=jnp.int32)
    val_short = jnp.array(2, dtype=jnp.int32)
    val_idle = jnp.array(0, dtype=jnp.int32)
    
    signal = jnp.where(is_long, val_long, jnp.where(is_short, val_short, val_idle))
    
    return signal, box_high, box_low
import pytest
import jax
import jax.numpy as jnp
from packs.the_accountant.tqg_filters import analyze_signal_core

# Pakotetaan XLA-tarkkuus
jax.config.update("jax_enable_x64", True)

def create_base_tensor():
    """Apufunktio: Luo tyhjän nollatensorin (8, 3, 30, 4)"""
    return jnp.zeros((8, 3, 30, 4), dtype=jnp.float64)

def test_tqg_long_surgical_strike():
    """Skenaario 1: Täydellinen LONG signaali."""
    tensor = create_base_tensor()
    
    # GATE I: Volatiliteettipiikki
    tensor = tensor.at[0, 0, :-1, 1].set(1.1010) # Hist. High
    tensor = tensor.at[0, 0, :-1, 2].set(1.1000) # Hist. Low
    tensor = tensor.at[0, 0, -1, 1].set(1.1020)  # Nykyinen High
    tensor = tensor.at[0, 0, -1, 2].set(1.1000)  # Nykyinen Low
    
    # GATE II: Bullish Alignment (Close > Open)
    tensor = tensor.at[0, 0, -1, 0].set(1.1005) # M1 Open
    tensor = tensor.at[0, 0, -1, 3].set(1.1018) # M1 Close
    tensor = tensor.at[0, 1, -1, 0].set(1.1000) # M5 Open
    tensor = tensor.at[0, 1, -1, 3].set(1.1015) # M5 Close
    
    # GATE III: RNAI Ostoaggressio
    tensor = tensor.at[0, 2, -1, 0].set(1.5)
    
    signals, box_highs, box_lows = analyze_signal_core(tensor)
    assert signals[0] == 1  # 1 = LONG
    assert jnp.isclose(box_highs[0], 1.1020)

def test_tqg_short_surgical_strike():
    """Skenaario 2: Täydellinen SHORT signaali."""
    tensor = create_base_tensor()
    
    # GATE I: Volatiliteettipiikki
    tensor = tensor.at[0, 0, :-1, 1].set(1.1010)
    tensor = tensor.at[0, 0, :-1, 2].set(1.1000)
    tensor = tensor.at[0, 0, -1, 1].set(1.1020)
    tensor = tensor.at[0, 0, -1, 2].set(1.1000)
    
    # GATE II: Bearish Alignment (Close < Open)
    tensor = tensor.at[0, 0, -1, 0].set(1.1018) # M1 Open korkealla
    tensor = tensor.at[0, 0, -1, 3].set(1.1005) # M1 Close matalalla
    tensor = tensor.at[0, 1, -1, 0].set(1.1015) # M5 Open korkealla
    tensor = tensor.at[0, 1, -1, 3].set(1.1000) # M5 Close matalalla
    
    # GATE III: RNAI Myyntiaggressio (Negatiivinen)
    tensor = tensor.at[0, 2, -1, 0].set(-1.5)
    
    signals, box_highs, box_lows = analyze_signal_core(tensor)
    assert signals[0] == 2  # 2 = SHORT
    assert jnp.isclose(box_lows[0], 1.1000)

def test_tqg_gate1_rejection_low_volatility():
    """Skenaario 3: Hylkäys. Rakenne ja RNAI kunnossa, mutta ei volyymipiikkiä."""
    tensor = create_base_tensor()
    
    # GATE I: PIENI volatiliteetti (Nykyinen High-Low sama kuin historiassa)
    tensor = tensor.at[0, 0, :-1, 1].set(1.1010)
    tensor = tensor.at[0, 0, :-1, 2].set(1.1000)
    tensor = tensor.at[0, 0, -1, 1].set(1.1010)  # Ei piikkiä
    tensor = tensor.at[0, 0, -1, 2].set(1.1000)
    
    # GATE II & III: Täydelliset (Bullish)
    tensor = tensor.at[0, 0, -1, 0].set(1.1005)
    tensor = tensor.at[0, 0, -1, 3].set(1.1009)
    tensor = tensor.at[0, 1, -1, 0].set(1.1000)
    tensor = tensor.at[0, 1, -1, 3].set(1.1008)
    tensor = tensor.at[0, 2, -1, 0].set(1.5)
    
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0  # 0 = IDLE (Hylätty!)

def test_tqg_gate2_rejection_structural_mismatch():
    """Skenaario 4: Hylkäys. Volyymi ja RNAI kunnossa, mutta M1 ja M5 ovat eri mieltä."""
    tensor = create_base_tensor()
    
    # GATE I & III: Täydelliset (Volyymipiikki ja vahva RNAI)
    tensor = tensor.at[0, 0, :-1, 1].set(1.1010)
    tensor = tensor.at[0, 0, :-1, 2].set(1.1000)
    tensor = tensor.at[0, 0, -1, 1].set(1.1020)
    tensor = tensor.at[0, 0, -1, 2].set(1.1000)
    tensor = tensor.at[0, 2, -1, 0].set(1.5) # Ostoaggressio
    
    # GATE II: RISTIRIITA (M1 on Bullish, mutta M5 on Bearish)
    tensor = tensor.at[0, 0, -1, 0].set(1.1005)
    tensor = tensor.at[0, 0, -1, 3].set(1.1018) # M1 Bullish
    tensor = tensor.at[0, 1, -1, 0].set(1.1015)
    tensor = tensor.at[0, 1, -1, 3].set(1.1000) # M5 Bearish!
    
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0  # 0 = IDLE (Hylätty!)

def test_tqg_gate3_rejection_weak_rnai():
    """Skenaario 5: Hylkäys. Rakenne ja volyymi hyviä, mutta RNAI liian heikko."""
    tensor = create_base_tensor()
    
    # GATE I & II: Täydelliset (Volyymipiikki ja M1/M5 Bullish Alignment)
    tensor = tensor.at[0, 0, :-1, 1].set(1.1010)
    tensor = tensor.at[0, 0, :-1, 2].set(1.1000)
    tensor = tensor.at[0, 0, -1, 1].set(1.1020)
    tensor = tensor.at[0, 0, -1, 2].set(1.1000)
    tensor = tensor.at[0, 0, -1, 0].set(1.1005)
    tensor = tensor.at[0, 0, -1, 3].set(1.1018)
    tensor = tensor.at[0, 1, -1, 0].set(1.1000)
    tensor = tensor.at[0, 1, -1, 3].set(1.1015)
    
    # GATE III: HEIKKO RNAI (0.5, kun vaatimus on > 1.0)
    tensor = tensor.at[0, 2, -1, 0].set(0.5)
    
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0  # 0 = IDLE (Hylätty!)
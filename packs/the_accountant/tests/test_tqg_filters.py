import pytest
import jax
import jax.numpy as jnp
from packs.the_accountant.tqg_filters import analyze_signal_core

jax.config.update("jax_enable_x64", True)

def create_base_tensor():
    # 8 symbolia, 3 (M1, M5, RNAI), 30 kynttilää, 4 hintaa (OHLC)
    tensor = jnp.zeros((8, 3, 30, 4), dtype=jnp.float64)
    # Perusvolatiliteetti M1: High = 1.1010, Low = 1.1000 (Range 0.0010)
    tensor = tensor.at[:, 0, :, 1].set(1.1010)
    tensor = tensor.at[:, 0, :, 2].set(1.1000)
    return tensor

# 1. TÄYDELLINEN LONG
def test_tqg_long_vojker_strategy():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1040)
    tensor = tensor.at[0, 0, -1, 2].set(1.1000) # Vol 0.0040 > 1.5 * 0.0010
    tensor = tensor.at[0, 0, -1, 0].set(1.1000) # M1 Open
    tensor = tensor.at[0, 0, -1, 3].set(1.1020) # M1 Close (Vihreä)
    tensor = tensor.at[0, 1, -1, 0].set(1.1000) # M5 Open
    tensor = tensor.at[0, 1, -1, 3].set(1.1020) # M5 Close (Vihreä)
    tensor = tensor.at[0, 2, -1, 0].set(1.5) # RNAI > 1.0
    signals, box_highs, box_lows = analyze_signal_core(tensor)
    assert signals[0] == 1 
    assert jnp.isclose(box_highs[0], 1.1040)

# 2. TÄYDELLINEN SHORT
def test_tqg_short_vojker_strategy():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1010)
    tensor = tensor.at[0, 0, -1, 2].set(1.0970) # Vol 0.0040
    tensor = tensor.at[0, 0, -1, 0].set(1.1000) # M1 Open
    tensor = tensor.at[0, 0, -1, 3].set(1.0980) # M1 Close (Punainen)
    tensor = tensor.at[0, 1, -1, 0].set(1.1000) # M5 Open
    tensor = tensor.at[0, 1, -1, 3].set(1.0980) # M5 Close (Punainen)
    tensor = tensor.at[0, 2, -1, 0].set(-1.5) # RNAI < -1.0
    signals, box_highs, box_lows = analyze_signal_core(tensor)
    assert signals[0] == 2
    assert jnp.isclose(box_lows[0], 1.0970)

# 3. HYLKÄYS: Gate I (Matala volatiliteetti)
def test_tqg_rejection_gate1_low_vol():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1012)
    tensor = tensor.at[0, 0, -1, 2].set(1.1000) # Vol 0.0012 <= 1.5 * 0.0010
    tensor = tensor.at[0, 0, -1, 0].set(1.1000) 
    tensor = tensor.at[0, 0, -1, 3].set(1.1010) # M1 Vihreä
    tensor = tensor.at[0, 1, -1, 0].set(1.1000) 
    tensor = tensor.at[0, 1, -1, 3].set(1.1010) # M5 Vihreä
    tensor = tensor.at[0, 2, -1, 0].set(1.5) # RNAI OK
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0 # TÄYTYY HYLÄTÄ

# 4. HYLKÄYS: Gate II (M1 ja M5 erimieliset)
def test_tqg_rejection_gate2_mismatch():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1040)
    tensor = tensor.at[0, 0, -1, 2].set(1.1000) # Vol OK
    tensor = tensor.at[0, 0, -1, 0].set(1.1000) 
    tensor = tensor.at[0, 0, -1, 3].set(1.1020) # M1 Vihreä
    tensor = tensor.at[0, 1, -1, 0].set(1.1020) 
    tensor = tensor.at[0, 1, -1, 3].set(1.1000) # M5 Punainen! (Ristiriita)
    tensor = tensor.at[0, 2, -1, 0].set(1.5) # RNAI OK
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0 # TÄYTYY HYLÄTÄ

# 5. HYLKÄYS: Gate III (Heikko RNAI Long-suuntaan)
def test_tqg_rejection_gate3_weak_rnai_long():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1040)
    tensor = tensor.at[0, 0, -1, 2].set(1.1000) # Vol OK
    tensor = tensor.at[0, 0, -1, 0].set(1.1000) 
    tensor = tensor.at[0, 0, -1, 3].set(1.1020) # M1 Vihreä
    tensor = tensor.at[0, 1, -1, 0].set(1.1000) 
    tensor = tensor.at[0, 1, -1, 3].set(1.1020) # M5 Vihreä
    tensor = tensor.at[0, 2, -1, 0].set(0.5) # RNAI = 0.5 (Vaatimus > 1.0)
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0 # TÄYTYY HYLÄTÄ

# 6. HYLKÄYS: Gate III (Heikko RNAI Short-suuntaan)
def test_tqg_rejection_gate3_weak_rnai_short():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1010)
    tensor = tensor.at[0, 0, -1, 2].set(1.0970) # Vol OK
    tensor = tensor.at[0, 0, -1, 0].set(1.1000) 
    tensor = tensor.at[0, 0, -1, 3].set(1.0980) # M1 Punainen
    tensor = tensor.at[0, 1, -1, 0].set(1.1000) 
    tensor = tensor.at[0, 1, -1, 3].set(1.0980) # M5 Punainen
    tensor = tensor.at[0, 2, -1, 0].set(-0.5) # RNAI = -0.5 (Vaatimus < -1.0)
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0 # TÄYTYY HYLÄTÄ

# 7. ERISTYS: Vain yksi symboli saa signaalin
def test_tqg_isolation_between_symbols():
    tensor = create_base_tensor()
    # Vain symboli 0 triggeröityy (Long)
    tensor = tensor.at[0, 0, -1, 1].set(1.1040)
    tensor = tensor.at[0, 0, -1, 2].set(1.1000) 
    tensor = tensor.at[0, 0, -1, 0].set(1.1000) 
    tensor = tensor.at[0, 0, -1, 3].set(1.1020) 
    tensor = tensor.at[0, 1, -1, 0].set(1.1000) 
    tensor = tensor.at[0, 1, -1, 3].set(1.1020) 
    tensor = tensor.at[0, 2, -1, 0].set(1.5) 
    
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 1
    # Varmistetaan että symboli 1 on tyhjä, eikä signaali vuoda!
    assert signals[1] == 0
import pytest
import jax
import jax.numpy as jnp
from packs.the_accountant.tqg_filters import analyze_signal_core

jax.config.update("jax_enable_x64", True)

def create_base_tensor():
    """Luo laboratoriotensorin, jossa on looginen perusdata."""
    tensor = jnp.zeros((8, 3, 30, 4), dtype=jnp.float64)
    # Historia (0-28): Range 0.0010 (1.1010 - 1.1000)
    tensor = tensor.at[:, 0:2, :-1, 1].set(1.1010) # Highs
    tensor = tensor.at[:, 0:2, :-1, 2].set(1.1000) # Lows
    # Alustetaan Live-kynttilä [-1] oletuksena laatikon sisään (ei signaalia)
    tensor = tensor.at[:, 0:2, -1, 0:4].set(1.1005) 
    return tensor

# 1. TÄYDELLINEN LONG
def test_long_surgical_strike():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1030) # High
    tensor = tensor.at[0, 0, -1, 2].set(1.1000) # Low (Vola OK: 0.0030)
    tensor = tensor.at[0, 0, -1, 3].set(1.1020) # Close (BOS UP OK)
    tensor = tensor.at[0, 1, -1, 3].set(1.1020) # M5 BOS OK
    tensor = tensor.at[0, 2, -1, 0].set(0.8)    # RNAI OK
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 1

# 2. TÄYDELLINEN SHORT
def test_short_surgical_strike():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1010) # High
    tensor = tensor.at[0, 0, -1, 2].set(1.0970) # Low (Vola OK: 0.0040)
    tensor = tensor.at[0, 0, -1, 3].set(1.0980) # Close (BOS DOWN OK)
    tensor = tensor.at[0, 1, -1, 3].set(1.0980) # M5 BOS OK
    tensor = tensor.at[0, 2, -1, 0].set(-0.8)   # RNAI OK
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 2

# 3. HYLKÄYS: MATALA VOLATILITEETTI
def test_reject_low_volatility():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1012) # High
    tensor = tensor.at[0, 0, -1, 2].set(1.1000) # Low (Vola 0.0012 < 0.0015)
    tensor = tensor.at[0, 0, -1, 3].set(1.1011) # BOS yritetään
    tensor = tensor.at[0, 2, -1, 0].set(0.9)
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0

# 4. HYLKÄYS: RNAI LIIAN HEIKKO
def test_reject_weak_rnai():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1030)
    tensor = tensor.at[0, 0, -1, 2].set(1.1000)
    tensor = tensor.at[0, 0, -1, 3].set(1.1020) # BOS OK
    tensor = tensor.at[0, 1, -1, 3].set(1.1020) 
    tensor = tensor.at[0, 2, -1, 0].set(0.39)   # RNAI 0.39 < 0.40
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0

# 5. HYLKÄYS: M5 RISTIRIITA
def test_reject_m5_mismatch():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1030)
    tensor = tensor.at[0, 0, -1, 3].set(1.1020) # M1 BOS OK
    tensor = tensor.at[0, 1, -1, 3].set(1.1005) # M5 ei BOS
    tensor = tensor.at[0, 2, -1, 0].set(0.8)
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0

# 6. HYLKÄYS: RNAI VÄÄRÄ SUUNTA
def test_reject_wrong_rnai_direction():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1030)
    tensor = tensor.at[0, 0, -1, 3].set(1.1020) # Hinta nousee
    tensor = tensor.at[0, 2, -1, 0].set(-0.8)   # Mutta myyntipaine
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0

# 7. RAJATILA: TASAN REUNALLA
def test_reject_price_at_edge():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 3].set(1.1010) # Tasan laatikon huippu
    tensor = tensor.at[0, 2, -1, 0].set(0.8)
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 0

# 8. REPAINTING-VARMISTUS
def test_no_repainting():
    tensor = create_base_tensor()
    tensor = tensor.at[0, 0, -1, 1].set(1.1050) # Live huippu korkealla
    tensor = tensor.at[0, 0, -1, 3].set(1.1005) # Mutta sulkeutuu sisään
    signals, box_highs, _ = analyze_signal_core(tensor)
    assert signals[0] == 0
    assert jnp.isclose(box_highs[0], 1.1010) # Laatikko ei repaintannut!

# 9. MONEN SYMBOLIN ERISTYS
def test_multi_symbol_isolation():
    tensor = create_base_tensor()
    # Symboli 0: LONG
    tensor = tensor.at[0, 0, -1, 1].set(1.1030)
    tensor = tensor.at[0, 0, -1, 2].set(1.1000)
    tensor = tensor.at[0, 0, -1, 3].set(1.1020)
    tensor = tensor.at[0, 1, -1, 3].set(1.1020)
    tensor = tensor.at[0, 2, -1, 0].set(0.8)
    # Symboli 7: SHORT
    tensor = tensor.at[7, 0, -1, 1].set(1.1010) # High
    tensor = tensor.at[7, 0, -1, 2].set(1.0970) # Low
    tensor = tensor.at[7, 0, -1, 3].set(1.0980) # Close
    tensor = tensor.at[7, 1, -1, 3].set(1.0980)
    tensor = tensor.at[7, 2, -1, 0].set(-0.8)
    
    signals, _, _ = analyze_signal_core(tensor)
    assert signals[0] == 1
    assert signals[7] == 2
    assert signals[4] == 0

# 10. NOLLA-VOLA SUOJA
def test_zero_volatility_safety():
    tensor = jnp.zeros((8, 3, 30, 4), dtype=jnp.float64)
    signals, _, _ = analyze_signal_core(tensor)
    assert jnp.all(signals == 0)
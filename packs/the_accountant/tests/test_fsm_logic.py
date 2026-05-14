import pytest
import jax
import jax.numpy as jnp
from packs.the_accountant.fsm_logic import master_fsm_step
from packs.the_accountant.config import GET_LEVERAGED_50K

jax.config.update("jax_enable_x64", True)

def test_master_fsm_scenarios():
    config_tensor = GET_LEVERAGED_50K.to_jax_array()
    
    # Batch skenaariot:
    # 0: Skenaario A - Täydellinen osuma (Kaikki OK) -> State 3
    # 1: Skenaario B - Jidoka-giljotiini (Signaali hyvä, mutta päivätappio ylittynyt) -> State 0
    
    # Luodaan mock symbol_tensor (M1, M5, RNAI)
    # TQG-vaatimukset: Vola 1.5x, Alignment, RNAI > 1.0
    symbol_tensors = jnp.zeros((2, 3, 30, 4), dtype=jnp.float64)
    # Asetetaan historia
    symbol_tensors = symbol_tensors.at[:, 0:2, :-1, 1].set(1.1010) # Highs
    symbol_tensors = symbol_tensors.at[:, 0:2, :-1, 2].set(1.1000) # Lows
    # Asetetaan live trigger (Long)
    symbol_tensors = symbol_tensors.at[:, 0, -1, 1].set(1.1040) # High
    symbol_tensors = symbol_tensors.at[:, 0, -1, 2].set(1.1000) # Low
    symbol_tensors = symbol_tensors.at[:, 0, -1, 3].set(1.1020) # Close (BOS)
    symbol_tensors = symbol_tensors.at[:, 1, -1, 3].set(1.1020) # M5 Close
    symbol_tensors = symbol_tensors.at[:, 2, -1, 0].set(1.5)    # RNAI
    
    current_states = jnp.array([1, 1], dtype=jnp.int32)
    equities = jnp.array([50000.0, 49000.0], dtype=jnp.float64)
    hwms = jnp.array([50000.0, 50000.0], dtype=jnp.float64)
    
    # Daily PnL: A=0.0 (Turvassa), B=-0.016 (Rikki, raja -0.015)
    daily_pnls = jnp.array([0.0, -0.016], dtype=jnp.float64)
    
    vmap_master = jax.vmap(master_fsm_step, in_axes=(0, 0, 0, 0, 0, None))
    f_states, f_signals, f_volumes = vmap_master(current_states, symbol_tensors, equities, hwms, daily_pnls, config_tensor)
    
    # Skenaario A: Kaikki OK
    assert f_states[0] == 3
    assert f_signals[0] == 1
    assert f_volumes[0] > 0
    
    # Skenaario B: Jidoka-giljotiini (The Ultimate Override)
    # Vaikka TQG antaa signaalin, Jidoka pakottaa tilaksi 0 ja maskaa ulostulot.
    assert f_states[1] == 0
    assert f_signals[1] == 0
    assert f_volumes[1] == 0.0
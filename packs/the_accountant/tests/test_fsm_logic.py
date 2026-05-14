import pytest
import jax
import jax.numpy as jnp
from packs.the_accountant.fsm_logic import evaluate_jidoka_daily
from packs.the_accountant.config import GET_LEVERAGED_50K

# Pakotetaan 64-bit tarkkuus myös testiajoon (NotebookLM:n vaatimus)
jax.config.update("jax_enable_x64", True)

def test_jidoka_safe_zone():
    """Skenaario: Päiväsaldo on turvassa (-1.0%). FSM saa jatkaa tilassaan (2: ARMED)."""
    current_state = jnp.array(2, dtype=jnp.int32)
    daily_pnl_pct = jnp.array(-0.010, dtype=jnp.float64)
    config_tensor = GET_LEVERAGED_50K.to_jax_array()
    
    new_state = evaluate_jidoka_daily(current_state, daily_pnl_pct, config_tensor)
    assert new_state == 2

def test_jidoka_danger_zone():
    """Skenaario: Päiväsaldo osuu hätärajaan (-1.5%). FSM pakotetaan tilaan 0 (NEUTRAL)."""
    current_state = jnp.array(3, dtype=jnp.int32) # Vaikka oltaisiin 3: ACTION -tilassa
    daily_pnl_pct = jnp.array(-0.015, dtype=jnp.float64)
    config_tensor = GET_LEVERAGED_50K.to_jax_array()
    
    new_state = evaluate_jidoka_daily(current_state, daily_pnl_pct, config_tensor)
    assert new_state == 0

def test_jidoka_vmap_compatibility():
    """
    Cpk 3.0 Vaatimus: Vektorisointi.
    Testataan 4 skenaariota kerralla (batch_size=4) ilman for-silmukoita.
    """
    current_states = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
    daily_pnls = jnp.array([0.05, -0.01, -0.016, -0.05], dtype=jnp.float64)
    config_tensor = GET_LEVERAGED_50K.to_jax_array()
    
    # jax.vmap: Määritellään, että 1. ja 2. argumentti ovat vektoreita (0-akseli),
    # mutta 3. argumentti (config_tensor) on vakio kaikille (None).
    vmap_jidoka = jax.vmap(evaluate_jidoka_daily, in_axes=(0, 0, None))
    
    new_states = vmap_jidoka(current_states, daily_pnls, config_tensor)
    
    # Odotetut tilat: 1 (turva), 2 (turva), 0 (hätäseis), 0 (hätäseis)
    assert jnp.array_equal(new_states, jnp.array([1, 2, 0, 0], dtype=jnp.int32))
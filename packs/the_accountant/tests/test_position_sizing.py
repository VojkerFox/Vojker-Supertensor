import pytest
import jax
import jax.numpy as jnp
from packs.the_accountant.fsm_logic import validate_and_size_position
from packs.the_accountant.config import GET_LEVERAGED_50K

# Cpk 3.0 Standardi
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def config():
    return GET_LEVERAGED_50K.to_jax_array()

def test_sizing_standard_long(config):
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.1010), jnp.array(1.1010), jnp.array(1.1000), config
    )
    assert final_sig == 1
    assert jnp.isclose(vol, 2.5)

def test_sizing_standard_short(config):
    final_sig, vol = validate_and_size_position(
        jnp.array(2), jnp.array(1.1000), jnp.array(1.1020), jnp.array(1.1000), config
    )
    assert final_sig == 2
    assert jnp.isclose(vol, 1.25)

def test_sizing_reject_min_lot(config):
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.4000), jnp.array(1.4000), jnp.array(1.1000), config
    )
    assert final_sig == 0
    assert vol == 0.0

def test_sizing_reject_tiny_sl(config):
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.1001), jnp.array(1.1001), jnp.array(1.1000), config
    )
    assert final_sig == 0

def test_sizing_idle_remains_idle(config):
    final_sig, vol = validate_and_size_position(
        jnp.array(0), jnp.array(1.1000), jnp.array(1.1010), jnp.array(1.0990), config
    )
    assert final_sig == 0
    assert vol == 0.0

def test_sizing_zero_sl_protection(config):
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.1000), jnp.array(1.1000), jnp.array(1.1000), config
    )
    assert final_sig == 0
    assert vol == 0.0

def test_sizing_negative_sl_protection(config):
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.0990), jnp.array(1.1000), jnp.array(1.1000), config
    )
    assert final_sig == 0

def test_sizing_exact_min_lot(config):
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.1250), jnp.array(1.1250), jnp.array(1.1000), config
    )
    assert final_sig == 1
    assert jnp.isclose(vol, 0.10)

def test_sizing_exact_min_sl(config):
    # Testaa rajatilaa (3 pipsiä) Epsilon-korjauksen kanssa
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.1003), jnp.array(1.1003), jnp.array(1.1000), config
    )
    assert final_sig == 1

def test_sizing_vmap_batch(config):
    vmap_size = jax.vmap(validate_and_size_position, in_axes=(0, 0, 0, 0, None))
    signals = jnp.array([1, 0, 2], dtype=jnp.int32)
    prices = jnp.array([1.1010, 1.1000, 1.0990], dtype=jnp.float64)
    highs = jnp.array([1.1010, 1.1000, 1.1010], dtype=jnp.float64)
    lows = jnp.array([1.1000, 1.1000, 1.1000], dtype=jnp.float64)
    
    sigs, vols = vmap_size(signals, prices, highs, lows, config)
    assert sigs[0] == 1
    assert sigs[2] == 2
    assert jnp.isclose(vols[2], 1.25)
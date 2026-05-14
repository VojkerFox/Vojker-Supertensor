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

# 1. STANDARDI LONG OK
def test_sizing_standard_long(config):
    # SL etäisyys 10 pipsiä (0.0010)
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.1010), jnp.array(1.1010), jnp.array(1.1000), config
    )
    assert final_sig == 1
    assert jnp.isclose(vol, 2.5) # 250 / (0.0010 * 100000)

# 2. STANDARDI SHORT OK
def test_sizing_standard_short(config):
    # SL etäisyys 20 pipsiä (0.0020)
    final_sig, vol = validate_and_size_position(
        jnp.array(2), jnp.array(1.1000), jnp.array(1.1020), jnp.array(1.1000), config
    )
    assert final_sig == 2
    assert jnp.isclose(vol, 1.25) # 250 / (0.0020 * 100000)

# 3. HYLKÄYS: MINIMILOTI ALITTUU (MASSIVE SL)
def test_sizing_reject_min_lot(config):
    # SL etäisyys 3000 pipsiä (0.3000) -> Volyymi 0.0083... < 0.01
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.4000), jnp.array(1.4000), jnp.array(1.1000), config
    )
    assert final_sig == 0
    assert vol == 0.0

# 4. HYLKÄYS: SL LIIAN PIENI (SLIPPAGE RISKI)
def test_sizing_reject_tiny_sl(config):
    # SL etäisyys 1 pip (0.0001) < 3 pipsin raja
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.1001), jnp.array(1.1001), jnp.array(1.1000), config
    )
    assert final_sig == 0

# 5. IDLE PYSYY IDLENÄ
def test_sizing_idle_remains_idle(config):
    final_sig, vol = validate_and_size_position(
        jnp.array(0), jnp.array(1.1000), jnp.array(1.1010), jnp.array(1.0990), config
    )
    assert final_sig == 0
    assert vol == 0.0

# 6. SUOJA: NOLLA-ETÄISYYS (NaN PREVENTION)
def test_sizing_zero_sl_protection(config):
    # SL etäisyys tasan 0 (box_low == live_price)
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.1000), jnp.array(1.1000), jnp.array(1.1000), config
    )
    assert final_sig == 0
    assert vol == 0.0 # Ei NaN-arvoa, koska jnp.inf-jako

# 7. SUOJA: NEGATIIVINEN ETÄISYYS (FYSIKAN VASTAINEN)
def test_sizing_negative_sl_protection(config):
    # Hinta on jo stopin alapuolella
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.0990), jnp.array(1.1000), jnp.array(1.1000), config
    )
    assert final_sig == 0

# 8. RAJATILA: TARKKA MINIMILOTI
def test_sizing_exact_min_lot(config):
    # SL 250 pipsiä (0.0250) -> 250 / (0.0250 * 100000) = tasan 0.10 lots
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.1250), jnp.array(1.1250), jnp.array(1.1000), config
    )
    assert final_sig == 1
    assert jnp.isclose(vol, 0.10)

# 9. RAJATILA: TARKKA MIN_SL_DISTANCE
def test_sizing_exact_min_sl(config):
    # Raja on 3 pipsiä (0.00030)
    final_sig, vol = validate_and_size_position(
        jnp.array(1), jnp.array(1.1003), jnp.array(1.1003), jnp.array(1.1000), config
    )
    assert final_sig == 1 # Tasan rajalla hyväksytään

# 10. VMAP BATCH-ERISTYS
def test_sizing_vmap_batch(config):
    vmap_size = jax.vmap(validate_and_size_position, in_axes=(0, 0, 0, 0, None))
    signals = jnp.array([1, 0, 2], dtype=jnp.int32)
    prices = jnp.array([1.1010, 1.1000, 1.0990], dtype=jnp.float64)
    highs = jnp.array([1.1010, 1.1000, 1.1010], dtype=jnp.float64)
    lows = jnp.array([1.1000, 1.1000, 1.1000], dtype=jnp.float64) # Case 2 SL 20 pipsiä
    
    sigs, vols = vmap_size(signals, prices, highs, lows, config)
    assert sigs[0] == 1
    assert sigs[1] == 0
    assert sigs[2] == 2
    assert jnp.isclose(vols[2], 1.25)
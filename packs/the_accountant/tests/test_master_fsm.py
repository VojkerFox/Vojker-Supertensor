import pytest
import jax
import jax.numpy as jnp
from packs.the_accountant.fsm_logic import master_fsm_step
from packs.the_accountant.config import GET_LEVERAGED_50K

jax.config.update("jax_enable_x64", True)

@pytest.fixture
def base_setup():
    config = GET_LEVERAGED_50K.to_jax_array()
    tvm = 100000.0 # Standard Contract
    return config, tvm

def create_master_tensor(signal_type="long", live_close=1.1020, box_low=1.1010, box_high=1.1010, vol_avg=0.0010):
    """Luo täydellisen kalibroidun ohjaustensorin."""
    tensor = jnp.full((3, 30, 4), 1.1005, dtype=jnp.float64)
    # Volatiliteetti-historia
    tensor = tensor.at[0, :27, 1].set(1.1000 + vol_avg)
    tensor = tensor.at[0, :27, 2].set(1.1000)
    # Salaman Laatikko historiaan
    tensor = tensor.at[0, -4:-1, 1].set(box_high)
    tensor = tensor.at[0, -4:-1, 2].set(box_low)

    if signal_type == "long":
        tensor = tensor.at[0, -1, 1].set(live_close + 0.0010)
        tensor = tensor.at[0, -1, 2].set(box_low)
        tensor = tensor.at[0, -1, 3].set(live_close)
        tensor = tensor.at[1, -1, 3].set(live_close)
        tensor = tensor.at[2, -1, 0].set(1.5)
    elif signal_type == "short":
        tensor = tensor.at[0, -1, 1].set(box_high)
        tensor = tensor.at[0, -1, 2].set(live_close - 0.0010)
        tensor = tensor.at[0, -1, 3].set(live_close)
        tensor = tensor.at[1, -1, 3].set(live_close)
        tensor = tensor.at[2, -1, 0].set(-1.5)
    return tensor

# TESTIT 1-10
def test_m1_success_long(base_setup):
    c, t = base_setup
    res = master_fsm_step(1, create_master_tensor("long", 1.1020, 1.1010), 50000.0, 50000.0, 0.0, c, t)
    assert res[0] == 3 and jnp.isclose(res[2], 2.5)

def test_m2_success_short(base_setup):
    c, t = base_setup
    res = master_fsm_step(1, create_master_tensor("short", 1.1000, box_high=1.1020), 50000.0, 50000.0, 0.0, c, t)
    assert res[0] == 3 and jnp.isclose(res[2], 1.25)

def test_m3_daily_breach(base_setup):
    c, t = base_setup
    res = master_fsm_step(1, create_master_tensor("long"), 49000.0, 50000.0, -0.016, c, t)
    assert res[0] == 0

def test_m4_trailing_breach(base_setup):
    c, t = base_setup
    # Equity 46k, HWM 50k -> -8% breach
    res = master_fsm_step(1, create_master_tensor("long"), 46000.0, 50000.0, 0.0, c, t)
    assert res[0] == 0

def test_m5_sticky_neutral(base_setup):
    c, t = base_setup
    res = master_fsm_step(0, create_master_tensor("long"), 50000.0, 50000.0, 0.0, c, t)
    assert res[0] == 0

def test_m6_slippage_reject(base_setup):
    c, t = base_setup
    # 2.5 pips dist
    res = master_fsm_step(1, create_master_tensor("long", 1.10125, 1.1010), 50000.0, 50000.0, 0.0, c, t)
    assert res[0] == 1

def test_m7_minlot_reject(base_setup):
    c, t = base_setup
    # SL dist 5000 pips -> vol < 0.01
    res = master_fsm_step(1, create_master_tensor("long", 1.6010, 1.1010), 50000.0, 50000.0, 0.0, c, t)
    assert res[0] == 1

def test_m8_zero_dist_safety(base_setup):
    c, t = base_setup
    res = master_fsm_step(1, create_master_tensor("long", 1.1010, 1.1010), 50000.0, 50000.0, 0.0, c, t)
    assert res[0] == 1 and res[2] == 0.0

def test_m9_low_vol_reject(base_setup):
    c, t = base_setup
    res = master_fsm_step(1, create_master_tensor("long", vol_avg=0.0100), 50000.0, 50000.0, 0.0, c, t)
    assert res[0] == 1

def test_m10_vmap_batch_audit(base_setup):
    c, t = base_setup
    v_f = jax.vmap(master_fsm_step, in_axes=(0, 0, 0, 0, 0, None, 0))
    states = jnp.array([1, 1])
    tensors = jnp.stack([create_master_tensor("long"), create_master_tensor("long")])
    eqs = jnp.array([50000.0, 40000.0]) # Toinen tili tuhoutunut
    hwms = jnp.array([50000.0, 50000.0])
    pnls = jnp.array([0.0, 0.0])
    tvms = jnp.array([100000.0, 100000.0])
    f_states, _, _ = v_f(states, tensors, eqs, hwms, pnls, c, tvms)
    assert f_states[0] == 3 and f_states[1] == 0
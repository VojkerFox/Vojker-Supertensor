import pytest
import jax
import jax.numpy as jnp
from packs.the_accountant.fsm_logic import master_fsm_step
from packs.the_accountant.config import GET_LEVERAGED_50K

def test_nan_poisoning_poka_yoke():
    config = GET_LEVERAGED_50K.to_jax_array()
    tvm = 100000.0
    
    # Luodaan korruptoitunut tensori, jossa live-hinta on NaN
    poisoned_tensor = jnp.full((3, 30, 4), 1.1005, dtype=jnp.float64)
    poisoned_tensor = poisoned_tensor.at[0, -1, 3].set(jnp.nan) # MYRKKY
    
    # Suoritetaan askel
    # JAXissa NaN-vertailut (NaN > 0) palauttavat False. 
    # TQG-logiikkamme pitäisi palauttaa signal=0.
    state, sig, vol = master_fsm_step(1, poisoned_tensor, 50000.0, 50000.0, 0.0, config, tvm)
    
    # Tulos: Jos hinta on NaN, signaalin on oltava 0 ja tilan IDLE (tai NEUTRAL)
    assert jnp.isnan(vol) == False
    assert sig == 0
    assert state != 3 # Ei saa mennä ACTION-tilaan
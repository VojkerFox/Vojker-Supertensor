import pytest
import jax.numpy as jnp
from dataclasses import FrozenInstanceError
from packs.the_accountant.config import PropFirmConfig

def test_prop_firm_constants_are_accurate():
    """Varmistaa, että prop-firman perussäännöt ovat oikein."""
    config = PropFirmConfig()
    
    assert config.STARTING_BALANCE == 50000.0
    assert config.PROFIT_TARGET_PCT == 0.06
    assert config.MAX_DAILY_LOSS_PCT == -0.015
    assert config.MAX_TRAILING_LOSS_PCT == -0.06  # NOTEBOOKLM KORJAUS

def test_jidoka_monetary_limits():
    """Varmistaa, että Jidoka-hätäkatkaisimen absoluuttinen dollariarvo on deterministinen."""
    config = PropFirmConfig()
    
    expected_daily_loss = 50000.0 * -0.015
    assert config.daily_loss_limit_usd == expected_daily_loss
    assert config.daily_loss_limit_usd == -750.0

def test_config_is_immutable():
    """Cpk 3.0 vaatimus: Estää 'Black Box' -vuodot."""
    config = PropFirmConfig()
    
    with pytest.raises(FrozenInstanceError):
        config.STARTING_BALANCE = 100000.0

def test_jax_tensor_conversion():
    """Varmistaa JAX-tensorin muodon ja 64-bittisen tarkkuuden."""
    config = PropFirmConfig()
    tensor = config.to_jax_array()
    
    # Nyt vektorissa on 3 arvoa (Target, Daily, Trailing)
    assert tensor.shape == (3,) 
    # Varmistetaan 64-bittinen tarkkuus finanssilaskentaan
    assert tensor.dtype == jnp.float64 
    assert jnp.allclose(tensor, jnp.array([0.06, -0.015, -0.06], dtype=jnp.float64))
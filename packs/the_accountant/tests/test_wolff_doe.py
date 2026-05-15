import pytest
import jax.numpy as jnp
from packs.the_accountant.sniffer import run_full_factorial_doe

def test_full_factorial_grid_size():
    # Simuloidaan 10 instrumenttia
    batch_size = 10
    symbols = jnp.zeros((batch_size, 3, 30, 4))
    states = jnp.ones(batch_size, dtype=jnp.int32)
    tvms = jnp.full(batch_size, 100000.0)
    
    grid, results = run_full_factorial_doe(symbols, states, tvms)
    
    # Varmistetaan, että kokeita on tasan 27 (3x3x3)
    assert grid.shape[0] == 27
    assert results.shape[0] == 27
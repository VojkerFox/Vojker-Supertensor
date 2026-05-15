# -*- coding: utf-8 -*-
import pytest
import jax.numpy as jnp
from leveraged_core.jidoka_fsm import jidoka_daily_loss_cell

def test_jidoka_armed_state_normal_conditions():
    state, available_risk = jidoka_daily_loss_cell(
        daily_start_balance=jnp.float32(50000.0),
        current_equity=jnp.float32(49500.0),
        max_daily_loss_pct=jnp.float32(-0.03),
        vojker_buffer_pct=jnp.float32(-0.015)
    )
    assert state == jnp.int32(1)
    assert available_risk == pytest.approx(250.0, 0.1)

def test_jidoka_halt_state_at_buffer():
    state, available_risk = jidoka_daily_loss_cell(
        daily_start_balance=jnp.float32(50000.0),
        current_equity=jnp.float32(49250.0),
        max_daily_loss_pct=jnp.float32(-0.03),
        vojker_buffer_pct=jnp.float32(-0.015)
    )
    assert state == jnp.int32(0)
    assert available_risk == 0.0

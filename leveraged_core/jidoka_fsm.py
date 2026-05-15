# -*- coding: utf-8 -*-
import jax.numpy as jnp
from typing import Tuple

def jidoka_daily_loss_cell(
    daily_start_balance: jnp.float32,
    current_equity: jnp.float32,
    max_daily_loss_pct: jnp.float32,
    vojker_buffer_pct: jnp.float32
) -> Tuple[jnp.int32, jnp.float32]:
    start_bal = jnp.asarray(daily_start_balance, dtype=jnp.float32)
    equity = jnp.asarray(current_equity, dtype=jnp.float32)
    buffer_pct = jnp.asarray(vojker_buffer_pct, dtype=jnp.float32)
    
    max_allowed_loss_usd = start_bal * jnp.abs(buffer_pct)
    current_loss_usd = start_bal - equity
    
    available_risk = jnp.maximum(0.0, max_allowed_loss_usd - current_loss_usd)
    fsm_state = jnp.where(available_risk > 0.0, jnp.int32(1), jnp.int32(0))
    
    return fsm_state, available_risk

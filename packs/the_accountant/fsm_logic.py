# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp

# Pakotetaan 64-bittinen tarkkuus finanssilaskentaan
jax.config.update("jax_enable_x64", True)

@jax.jit
def evaluate_absolute_jidoka(current_state, current_equity, hwm, daily_pnl_pct, config_tensor):
    """Issue #3: Yhdistetty Jidoka-kilpi (Daily + Trailing)."""
    max_daily_loss = config_tensor[1]
    max_trailing_loss = config_tensor[2]
    
    trailing_pnl_pct = (current_equity - hwm) / hwm
    
    is_daily_breach = daily_pnl_pct <= max_daily_loss
    is_trailing_breach = trailing_pnl_pct <= max_trailing_loss
    
    # Bittitason OR-maski
    is_breach = is_daily_breach | is_trailing_breach
    
    return jnp.where(is_breach, 0, current_state)

@jax.jit
def validate_and_size_position(signal, live_price, box_high, box_low, config_tensor):
    """
    Issue #5: Deterministinen positiokoon laskenta numeerisella vakaudella.
    config_tensor: [..., FIXED_RISK_USD, MIN_SL_DISTANCE, CONTRACT_SIZE]
    """
    risk_usd = config_tensor[3]
    min_sl_dist = config_tensor[4]
    contract_size = config_tensor[5]
    
    # Epsilon-puskuri (1e-11) estää liukulukuvirheet (kuten 1.24999 vs 1.25)
    eps = 1e-11

    # 1. Laske SL-etäisyys (Käytetään jnp.inf NaN-vuodon estämiseen)
    sl_dist_long = live_price - box_low
    sl_dist_short = box_high - live_price

    sl_dist = jnp.where(signal == 1, sl_dist_long,
              jnp.where(signal == 2, sl_dist_short, jnp.inf))

    # 2. Laske raakavolyymi
    raw_volume = risk_usd / (sl_dist * contract_size)

    # 3. Poka-Yoke Hylkäysmaskit (Lisätään epsilon vertailuun)
    is_valid_vol = (raw_volume + eps) >= 0.01
    is_safe_dist = (sl_dist + eps) >= min_sl_dist
    
    is_safe = (signal > 0) & is_valid_vol & is_safe_dist
    
    # 4. Finalisointi: Lisätään eps ennen floor-pyöristystä
    final_signal = jnp.where(is_safe, signal, 0)
    final_volume = jnp.where(is_safe, jnp.floor((raw_volume + eps) * 100) / 100, 0.0)
    
    return final_signal, final_volume
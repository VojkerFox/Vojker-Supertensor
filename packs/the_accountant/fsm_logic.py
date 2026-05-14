# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
from packs.the_accountant.tqg_filters import process_symbol_logic

# Cpk 3.0 Standardi: Pakotetaan 64-bittinen tarkkuus
jax.config.update("jax_enable_x64", True)

def evaluate_absolute_jidoka(proposed_state, current_state, current_equity, hwm, daily_pnl_pct, config_tensor):
    """
    ISSUE #3: Absolute Jidoka Shield.
    Override: Pakottaa tilan nollaan (NEUTRAL) sääntörikkomuksesta TAI jos tila on jo 0.
    """
    max_daily_loss = config_tensor[1]
    max_trailing_loss = config_tensor[2]
    
    trailing_pnl_pct = (current_equity - hwm) / hwm
    is_daily_breach = daily_pnl_pct <= max_daily_loss
    is_trailing_breach = trailing_pnl_pct <= max_trailing_loss
    
    # Jidoka-giljotiini: Jos tili on kerran kuollut (0), se pysyy kuolleena (Sticky State)
    is_dead = (current_state == 0) | is_daily_breach | is_trailing_breach
    
    return jnp.where(is_dead, jnp.int32(0), proposed_state)

def validate_and_size_position(signal, live_price, box_high, box_low, config_tensor, tick_value_mult):
    """
    ISSUE #5: Poka-Yoke Position Sizing.
    Laskee volyymin ja hylkää vaaralliset kaupat (Slippage/MinLot).
    """
    risk_usd = config_tensor[3]
    min_sl_dist = config_tensor[4]
    contract_size = config_tensor[5]
    eps = 1e-11 # Numeerinen vakauspuskuri

    # SL-etäisyyden laskenta fysiikan mukaan
    sl_dist_long = live_price - box_low
    sl_dist_short = box_high - live_price
    sl_dist = jnp.where(signal == 1, sl_dist_long, jnp.where(signal == 2, sl_dist_short, jnp.inf))

    # Volyymikaava:
    # $$Volume = \frac{Risk\_USD}{SL\_Dist \times Contract\_Size}$$
    raw_volume = risk_usd / (sl_dist * contract_size)
    
    # Validointimaskit
    is_valid_vol = (raw_volume + eps) >= 0.01
    is_safe_dist = (sl_dist + eps) >= min_sl_dist
    is_safe = (signal > 0) & is_valid_vol & is_safe_dist
    
    # Lopullinen signaali ja pyöristetty volyymi
    final_signal = jnp.where(is_safe, signal, 0)
    final_volume = jnp.where(is_safe, jnp.floor((raw_volume + eps) * 100) / 100, 0.0)
    
    return final_signal, final_volume

@jax.jit
def master_fsm_step(current_state, symbol_tensor, current_equity, hwm, daily_pnl_pct, config_tensor, tick_value_mult):
    """
    THE MASTER LOGIC: Panama Process Core Integration.
    Yhdistää TQG -> Sizing -> Jidoka Override -> Masking.
    """
    # 1. TQG-analyysi (Vojker Strategy)
    raw_signal, box_high, box_low = process_symbol_logic(symbol_tensor)
    live_price = symbol_tensor[0, -1, 3] # M1 Live Close

    # 2. Positiokoon ja turvallisuuden validointi (Poka-Yoke)
    safe_signal, volume = validate_and_size_position(raw_signal, live_price, box_high, box_low, config_tensor, tick_value_mult)

    # 3. Ehdotettu tila: Jos signaali on turvallinen, ehdota ACTION (3), muuten IDLE (1)
    proposed_state = jnp.where(safe_signal > 0, jnp.int32(3), jnp.int32(1))

    # 4. Jidoka Shield: Viimeinen sana (Override)
    final_state = evaluate_absolute_jidoka(proposed_state, current_state, current_equity, hwm, daily_pnl_pct, config_tensor)

    # 5. Maskaus: Puhdistetaan signaalit jos Jidoka tai Poka-Yoke hylkäsi
    is_action = final_state == 3
    final_signal_out = jnp.where(is_action, safe_signal, 0)
    final_volume_out = jnp.where(is_action, volume, 0.0)

    return final_state, final_signal_out, final_volume_out
import jax
import jax.numpy as jnp

@jax.jit
def evaluate_jidoka_daily(current_state: jnp.ndarray, daily_pnl_pct: jnp.ndarray, config_tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Jidoka-hätäkatkaisin.
    Ottaa sisään nykyisen tilan, päivän PnL:n ja prop-firman sääntötensorin.
    Palauttaa uuden tilan.
    
    config_tensorin rakenne: [PROFIT_TARGET_PCT, MAX_DAILY_LOSS_PCT, MAX_TRAILING_LOSS_PCT]
    Indeksi 1 on MAX_DAILY_LOSS_PCT (esim. -0.015)
    """
    
    # Erotetaan päivittäinen tappioraja tensorista
    max_daily_loss = config_tensor[1]
    
    # Poka-Yoke: Puhdas matriisimaski (jax.where).
    # Ehto: Onko päivän tappio pienempi tai yhtä suuri kuin raja (esim. -0.016 <= -0.015 -> True)
    # Jos True -> Tila on 0 (NEUTRAL / HÄTÄSEIS).
    # Jos False -> Säilytetään current_state.
    new_state = jnp.where(daily_pnl_pct <= max_daily_loss, 0, current_state)
    
    # Palautetaan uusi tensori (ei in-place mutaatiota)
    return new_state
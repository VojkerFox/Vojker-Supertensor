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

# Cpk 3.0 Vaatimus: Vektorisointi ja monistettavuus (vmap)

@jax.jit
def evaluate_absolute_jidoka(current_state: jnp.ndarray, current_equity: jnp.ndarray, hwm: jnp.ndarray, daily_pnl_pct: jnp.ndarray, config_tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Absolute Jidoka Shield (Cpk 3.0 Standard).
    Yhdistää sekä päivittäisen että High Water Mark (HWM) -pohjaisen Trailing Lossin.
    Jos jompikumpi ylittää sallitun varoitusrajan, FSM pakotetaan 0: NEUTRAL.
    
    config_tensor: [PROFIT_TARGET_PCT, MAX_DAILY_LOSS_PCT, MAX_TRAILING_LOSS_PCT]
    """
    max_daily_loss = config_tensor[1]
    max_trailing_loss = config_tensor[2]

    # 1. Laske Trailing etäisyys (Kuinka paljon ollaan alle huipun)
    trailing_pnl_pct = (current_equity - hwm) / hwm

    # 2. Evaluoi bittimaskit (Onko rajat rikottu?)
    # Huom: JAX käyttää bittioperaattoreita boolean-taulukoille.
    is_daily_breach = daily_pnl_pct <= max_daily_loss
    is_trailing_breach = trailing_pnl_pct <= max_trailing_loss

    # 3. Yhdistä ehdot Poka-Yoke-kilveksi. Jos jompikumpi sääntö paukkuu, arvo on True.
    # VAROITUS: Käytä `|`, älä Pythonin `or` -avainsanaa!
    is_breach = is_daily_breach | is_trailing_breach

    # 4. Pakota tila 0 (HÄTÄSEIS), muuten säilytä nykyinen tila
    new_state = jnp.where(is_breach, 0, current_state)

    return new_state
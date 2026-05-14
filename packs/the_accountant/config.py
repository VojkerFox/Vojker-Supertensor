from dataclasses import dataclass
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

@dataclass(frozen=True)
class PropFirmConfig:
    STARTING_BALANCE: float = 50000.0
    PROFIT_TARGET_PCT: float = 0.06
    MAX_DAILY_LOSS_PCT: float = -0.015
    MAX_TRAILING_LOSS_PCT: float = -0.06
    
    # ISSUE #5: Riskinhallinnan vakiot
    FIXED_RISK_USD: float = 250.0 # 0.5% $50k tilistä
    MIN_SL_DISTANCE: float = 0.00030 # 3 pipsiä (esim. EURUSD)
    CONTRACT_SIZE: float = 100000.0 # Standardi lot-koko
    
    @property
    def daily_loss_limit_usd(self) -> float:
        return self.STARTING_BALANCE * self.MAX_DAILY_LOSS_PCT
        
    def to_jax_array(self) -> jnp.ndarray:
        return jnp.array([
            self.PROFIT_TARGET_PCT, 
            self.MAX_DAILY_LOSS_PCT,
            self.MAX_TRAILING_LOSS_PCT,
            self.FIXED_RISK_USD,
            self.MIN_SL_DISTANCE,
            self.CONTRACT_SIZE
        ], dtype=jnp.float64)

GET_LEVERAGED_50K = PropFirmConfig()

@jax.jit
def validate_and_size_position(signal: jnp.ndarray, live_price: jnp.ndarray, box_high: jnp.ndarray, box_low: jnp.ndarray, config_tensor: jnp.ndarray) -> tuple:
    """
    Issue #5: Deterministinen positiokoon laskenta ja Jidoka-hylkäys.
    config_tensor indeksit: 3=Risk, 4=MinSL, 5=ContractSize
    """
    risk_usd = config_tensor[3]
    min_sl_dist = config_tensor[4]
    contract_size = config_tensor[5]

    # 1. Laske SL-etäisyys maskaamalla (Estetään NaN-vuoto jnp.inf:llä)
    sl_dist_long = live_price - box_low
    sl_dist_short = box_high - live_price

    # Valitaan etäisyys signaalin mukaan. Jos signaali on 0, etäisyys on ääretön.
    sl_dist = jnp.where(signal == 1, sl_dist_long,
              jnp.where(signal == 2, sl_dist_short, jnp.inf))

    # 2. Laske raakavolyymi (Lots). contract_size toimii kertoimena.
    # Jos sl_dist on inf, volume on 0.0.
    raw_volume = risk_usd / (sl_dist * contract_size)

    # 3. Poka-Yoke Hylkäysmaskit
    is_valid_vol = raw_volume >= 0.01
    is_safe_dist = sl_dist >= min_sl_dist
    
    # Toimeksianto on turvallinen vain jos signaali > 0 JA volyymi OK JA etäisyys OK
    is_safe = (signal > 0) & is_valid_vol & is_safe_dist
    
    # 4. FSM-Signaalin kumoaminen: Pakota 0 jos ei turvallinen
    final_signal = jnp.where(is_safe, signal, 0)
    
    # Pyöristetään volyymi 0.01 tarkkuuteen (MT5 vaatimus)
    final_volume = jnp.where(is_safe, jnp.floor(raw_volume * 100) / 100, 0.0)
    
    return final_signal, final_volume
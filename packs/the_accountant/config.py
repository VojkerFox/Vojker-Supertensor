from dataclasses import dataclass
import jax
import jax.numpy as jnp

# Pakotetaan 64-bittinen tarkkuus finanssilaskentaan (Cpk 3.0 vaatimus)
# Estää senttitason pyöristysvirheet, jotka voisivat laukaista prop-firman sääntörikkomuksen.
jax.config.update("jax_enable_x64", True)

@dataclass(frozen=True)
class PropFirmConfig:
    """
    Immutable Oracle: Prop Firm -säännöt (Get-Leveraged $50k Turbo).
    """
    STARTING_BALANCE: float = 50000.0
    PROFIT_TARGET_PCT: float = 0.06
    MAX_DAILY_LOSS_PCT: float = -0.015
    MAX_TRAILING_LOSS_PCT: float = -0.06  # NOTEBOOKLM KORJAUS
    
    @property
    def daily_loss_limit_usd(self) -> float:
        return self.STARTING_BALANCE * self.MAX_DAILY_LOSS_PCT
        
    def to_jax_array(self) -> jnp.ndarray:
        """
        Pakkaa prop-firman rajat tensoriksi vektorisoitua (vmap) riskianalyysiä varten.
        Shape: (3,)
        Format: float64
        """
        return jnp.array([
            self.PROFIT_TARGET_PCT, 
            self.MAX_DAILY_LOSS_PCT,
            self.MAX_TRAILING_LOSS_PCT
        ], dtype=jnp.float64)

GET_LEVERAGED_50K = PropFirmConfig()
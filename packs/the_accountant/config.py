from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class PropFirmConfig:
    """
    Immutable Oracle: Prop Firm -säännöt (Get-Leveraged $50k Turbo).
    Kaikki arvot on jäädytetty (@dataclass(frozen=True)) estämään ajonaikaiset 
    mutaatiot (Side Effects) ja varmistamaan yhteensopivuus JAX XLA -käännöksen 
    (static_argnums) kanssa.
    """
    STARTING_BALANCE: float = 50000.0
    PROFIT_TARGET_PCT: float = 0.06
    MAX_DAILY_LOSS_PCT: float = -0.015
    
    @property
    def daily_loss_limit_usd(self) -> float:
        """
        Palauttaa tarkan dollarimääräisen LCL (Lower Control Limit) arvon.
        Tämä on FSM:n Jidoka-hätäkatkaisimen kova raja (-$750).
        """
        return self.STARTING_BALANCE * self.MAX_DAILY_LOSS_PCT
        
    def to_jax_array(self) -> jnp.ndarray:
        """
        Pakkaa prop-firman rajat (target, daily_loss) tensoriksi vektorisoitua 
        (vmap) riskianalyysiä varten.
        Shape: (2,)
        """
        return jnp.array([self.PROFIT_TARGET_PCT, self.MAX_DAILY_LOSS_PCT], dtype=jnp.float32)

# Luodaan instanssi, jota voidaan tuoda muihin tiedostoihin
GET_LEVERAGED_50K = PropFirmConfig()
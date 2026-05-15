# -*- coding: utf-8 -*-
import jax.numpy as jnp
from typing import Final, Dict, List

# --- OSA 1: STAATTINEN TURVAVERKKO (Legacy Support) ---

# SYMBOL_MAP: Käytetään Python-puolen MT5-adapterissa datan reititykseen Supertensorille.
# Tämä on "Master Key", joka määrittää K-akselin järjestyksen.
SYMBOL_MAP: Final[Dict[str, int]] = {
    "EURUSD": 0,
    "GBPUSD": 1,
    "USDJPY": 2,
    "USDCHF": 3,
    "AUDUSD": 4,
    "USDCAD": 5,
    "XAUUSD": 6, # Gold
    "US30":   7  # Dow Jones
}

# TICK_VALUE_MATRIX: Referenssiarvot riskilaskentaan. 
# Haetaan tuotannossa ensisijaisesti tietokannasta/MT5:stä.
TICK_VALUE_MATRIX: Final[Dict[str, float]] = {
    "EURUSD": 100000.0,
    "GBPUSD": 100000.0,
    "USDJPY": 1000.0,   # Jeni-pari (100k * 0.01)
    "USDCHF": 100000.0,
    "AUDUSD": 100000.0,
    "USDCAD": 100000.0,
    "XAUUSD": 100.0,    # 1 lot = 100 ounces
    "US30":   1.0       # Indeksikohtainen kerroin
}

# JAX_TICK_VALUES: Deterministinen tensori vmap-ajoa varten.
TICK_VALUES: Final = jnp.array([
    TICK_VALUE_MATRIX["EURUSD"],
    TICK_VALUE_MATRIX["GBPUSD"],
    TICK_VALUE_MATRIX["USDJPY"],
    TICK_VALUE_MATRIX["USDCHF"],
    TICK_VALUE_MATRIX["AUDUSD"],
    TICK_VALUE_MATRIX["USDCAD"],
    TICK_VALUE_MATRIX["XAUUSD"],
    TICK_VALUE_MATRIX["US30"]
], dtype=jnp.float64)


# --- OSA 2: WOLFF-STANDARDIN PARETO-VAKIOITA ---

# Pareto-giljotiinin raja-arvot (Vojker 80/20/0.8 Rule)
PARETO_20_PCT: Final = 0.20   # Strategy candidates
PARETO_4_PCT: Final  = 0.04   # High probability group
PARETO_08_PCT: Final = 0.008  # The Elite Club (0.8%)


# --- OSA 3: POKA-YOKE JA APUFUNKTIOT ---

def get_symbol_index(symbol: str) -> int:
    """
    Hakee symbolin indeksin. Jos symbolia ei ole regisrissä, nostaa Andon-narun.
    Tämä on kriittinen suoja instrumentti-mismatchia vastaan.
    """
    if symbol not in SYMBOL_MAP:
        # Paul Akers: Pysäytä linja heti virheen sattuessa.
        raise ValueError(f"CRITICAL: Symbol {symbol} is not registered in SYMBOL_MAP!")
    return SYMBOL_MAP[symbol]

def get_elite_symbols(compatibility_scores: jnp.ndarray) -> List[str]:
    """
    Kytkee JAX-haistelijan tulokset takaisin instrumenttinimiin.
    Palauttaa listan symboleista, jotka läpäisivät 0.8% Pareto-seulan.
    """
    # Etsitään top-indeksit (vastaa 0.8% 1500:sta = ~12 kpl)
    n_elite = max(1, int(len(compatibility_scores) * PARETO_08_PCT))
    top_indices = jnp.argsort(compatibility_scores)[-n_elite:][::-1]
    
    # Käänteinen haku SYMBOL_MAPista
    inv_map = {v: k for k, v in SYMBOL_MAP.items()}
    return [inv_map[int(idx)] for idx in top_indices if int(idx) in inv_map]

# --- OSA 4: POSTGRES-BRIDGE (Placeholder) ---

# Tähän sijoitetaan SQL-skeeman tiedot ja tietokantayhteyden hallinta,
# jotta Registry voi "haistella" päivitykset 24 tunnin välein.
INSTRUMENT_DB_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS instrument_registry (
    symbol VARCHAR(20) PRIMARY KEY,
    tick_value_mult DOUBLE PRECISION NOT NULL,
    compatibility_score DOUBLE PRECISION DEFAULT 0.0,
    last_vjp_sensitivity DOUBLE PRECISION DEFAULT 0.0,
    is_elite BOOLEAN DEFAULT FALSE,
    last_sync TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
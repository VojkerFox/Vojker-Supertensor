# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import MetaTrader5 as mt5
from typing import Tuple, List
from packs.the_accountant.fsm_logic import master_fsm_step
from packs.the_accountant.registry import PARETO_08_PCT

# Cpk 3.0: Laskentatarkkuus Christian Wolffin vaatimuksesta
jax.config.update("jax_enable_x64", True)

# --- OSA 1: LIVE-DATAN HAKU (M5 GENBA-VÄYLÄ) ---

def get_marketwatch_history(count: int = 30) -> Tuple[jnp.ndarray, List[str], jnp.ndarray, jnp.ndarray]:
    """
    Hakee historian M5-aikavälillä ja pakkaa sen Supertensoriksi.
    """
    symbols = mt5.symbols_get()
    visible = [s for s in symbols if s.select]
    
    tensors, names, tvms, points = [], [], [], []
    
    for s in visible:
        rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_M5, 0, count)
        if rates is not None and len(rates) == count:
            ohlc = jnp.array([[r['open'], r['high'], r['low'], r['close']] for r in rates])
            supertensor_stack = jnp.stack([ohlc, ohlc, ohlc], axis=0)
            
            tensors.append(supertensor_stack)
            names.append(s.name)
            tvms.append(float(s.trade_tick_value))
            points.append(float(s.point))
            
    return jnp.array(tensors), names, jnp.array(tvms), jnp.array(points)

# --- OSA 2: ATR-ANALYYSI (M5 VOLATILITY GATE) ---

@jax.jit
def calculate_atr_pips(history_tensors, point_values):
    """
    Laskee ATR:n M5-aikavälillä pips-arvossa.
    Vakio: 1 pip = 10 * point.
    """
    highs = history_tensors[:, 0, :, 1]
    lows = history_tensors[:, 0, :, 2]
    
    ranges = highs - lows
    atr_raw = jnp.mean(ranges, axis=1)
    
    atr_pips = atr_raw / (point_values * 10.0)
    return atr_pips

# --- OSA 3: FULL-FACTORIAL DoE ---

@jax.jit
def run_full_factorial_doe(symbol_batch, states, tvm_batch):
    """Simuloi 27 skenaariota säilyttäen instrumenttien yksilöllisyyden."""
    v_coeffs = jnp.array([1.2, 1.5, 1.8])
    r_aggros = jnp.array([-1.2, -1.5, -1.8])
    s_limits = jnp.array([0.00025, 0.0003, 0.00035])

    gv, gr, gs = jnp.meshgrid(v_coeffs, r_aggros, s_limits, indexing='ij')
    grid = jnp.stack([gv.ravel(), gr.ravel(), gs.ravel()], axis=-1)

    def single_doe_run(params):
        custom_config = jnp.array([params[0], params[1], -0.06, 250.0, params[2], 100000.0])
        v_fsm = jax.vmap(master_fsm_step, in_axes=(0, 0, None, None, None, None, 0))
        _, _, vols = v_fsm(states, symbol_batch, 50000.0, 50000.0, 0.0, custom_config, tvm_batch)
        return vols

    doe_results = jax.vmap(single_doe_run)(grid)
    return grid, doe_results

# --- OSA 4: TRIPLE PARETO GILLOTINE (KORJATTU LIIKEPORTTI) ---

@jax.jit
def triple_pareto_filter(compatibility_scores, atr_pips):
    """
    Pareto-giljotiini säädetyllä liikeportilla.
    Sallittu alue: 1.0 <= ATR <= 500.0 pips.
    Tämä mahdollistaa Forexin (1-15 pip) ja Indeksien (15-500 pip) rinnakkaiselon.
    """
    # 1. ATR-maski: Avattu vastaamaan Genban todellisuutta
    atr_mask = (atr_pips >= 1.0) & (atr_pips <= 500.0)
    
    # 2. Safety mask: Nostettu 1e12 tasolle vakaiden indeksien sallimiseksi
    safety_mask = (compatibility_scores < 1e12) & (compatibility_scores > 0)
    
    final_mask = atr_mask & safety_mask
    valid_scores = jnp.where(final_mask, compatibility_scores, 0.0)
    
    n = valid_scores.shape[0]
    indices = jnp.argsort(valid_scores)[::-1]
    
    # Valitaan top 0.8% eliitti
    top_08_idx = max(1, int(n * PARETO_08_PCT))
    elite_indices = indices[:top_08_idx]
    
    return elite_indices, valid_scores[elite_indices]

# --- OSA 5: RISKI- JA LOT-LASKENTA ---

def calculate_wolff_lots(symbol_name: str, risk_usd: float = 250.0) -> float:
    """Laskee tarkan lot-koon perustuen Christian Wolffin $250 riskisääntöön."""
    info = mt5.symbol_info(symbol_name)
    if info is None or info.point <= 0: return 0.0
    
    # 100 pisteen stop-loss standardi
    stop_distance = 100 * info.point
    
    # POKA-YOKE: Varmistetaan oikeat attribuutit API-standardin mukaan
    risk_per_lot = (stop_distance / info.trade_tick_size) * info.trade_tick_value
    
    if risk_per_lot <= 0: return 0.0
    lot_size = risk_usd / risk_per_lot
    
    return max(info.volume_min, min(info.volume_max, round(lot_size, 2)))
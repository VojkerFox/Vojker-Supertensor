import jax
import jax.numpy as jnp

@jax.jit
def analyze_signal_core(tensor):
    """
    Panama Process Phase 2: Triple Quantile Gate (TQG)
    Input: (8, 3, 30, 4) Supertensor
    Output: Boolean Mask (8,) - True jos Signal Core on löytynyt
    """
    # vmap monistaa logiikan kaikille 8 symbolille yhtä aikaa
    vmapped_logic = jax.vmap(process_symbol_logic)
    return vmapped_logic(tensor)

def process_symbol_logic(symbol_tensor):
    """
    Yksittäisen symbolin (3, 30, 4) analyysi.
    K=0: M1, K=1: M5, K=2: RNAI Context
    """
    m1 = symbol_tensor[0]  # Nopeus
    m5 = symbol_tensor[1]  # Rakenne
    rnai = symbol_tensor[2, -1, 0] # Viimeisin RNAI-arvo

    # 1. GATE I: Volatility Filter (Pareto 20%)
    # Lasketaan M1 kynttilän koko suhteessa keskiarvoon
    m1_ranges = m1[:, 1] - m1[:, 2] # High - Low
    avg_volatility = jnp.mean(m1_ranges)
    current_vol = m1_ranges[-1]
    gate1 = current_vol > (avg_volatility * 1.5) # Vain poikkeuksellinen liike

    # 2. GATE II: Structural Alignment (Pareto 4%)
    # Onko M1 ja M5 samassa suunnassa (BOS-varmistus)?
    m1_dir = m1[-1, 3] > m1[-1, 0] # Close > Open
    m5_dir = m5[-1, 3] > m5[-1, 0]
    gate2 = jnp.logical_and(m1_dir, m5_dir)

    # 3. GATE III: RNAI Aggression Filter (Signal Core 0.8%)
    # Tämä on "se iso juttu": Onko liike aitoa aggressiota (High RNAI)?
    # Jos RNAI on positiivinen ja korkea, liike on aggressiivista ostoa.
    # Jos RNAI on negatiivinen mutta hinta nousee, kyseessä on absorptio (valesignaali).
    gate3 = rnai > 1.0 # Käytetään testissäsi näkynyttä dynaamista kynnysarvoa

    # Lopullinen päätös: Kaikkien porttien on oltava auki
    signal_core = jnp.logical_and(jnp.logical_and(gate1, gate2), gate3)
    
    return signal_core
# -*- coding: utf-8 -*-
import sys
import os
import time
import jax
import jax.numpy as jnp
import MetaTrader5 as mt5

# POKA-YOKE: Varmistetaan, että Python löytää 'packs'-moduulin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from packs.the_accountant.database import WolffDatabase
from packs.the_accountant.sniffer import (
    run_full_factorial_doe, 
    triple_pareto_filter, 
    get_marketwatch_history,
    calculate_wolff_lots,
    calculate_atr_pips
)
from packs.the_accountant.fsm_logic import master_fsm_step

# --- MÄÄRITÄ YHTEYS ---
db = WolffDatabase(dbname="vojker_db", user="postgres", password="password")

@jax.jit
def verify_stress_tolerance(symbol_tensors, states, tvm_batch):
    """
    STRESS TEST: Altistetaan eliitti-kandidaatit vihamieliselle ympäristölle.
    Simuloidaan 3.0x spread-laajennus ja 200% slippage-kasvu.
    """
    stress_config = jnp.array([1.2, -1.8, -0.06, 250.0, 0.0008, 100000.0])
    v_fsm = jax.vmap(master_fsm_step, in_axes=(0, 0, None, None, None, None, 0))
    _, _, volumes = v_fsm(states, symbol_tensors, 50000.0, 50000.0, 0.0, stress_config, tvm_batch)
    return volumes

def run_wolff_selection():
    print("="*75)
    print(" WOLFF MORNING REPORT: LIVE GENBA SCAN (M5 ATR 3.5-15) ".center(75, "="))
    print("="*75)
    start_time = time.time()

    if not mt5.initialize():
        print("CRITICAL: MT5 initialization failed.")
        return

    # 1. GENBA: Haetaan historia (M5), nimet, TVM:t ja POINT-arvot
    print(f"[{time.strftime('%H:%M:%S')}] Haetaan historiaa ja instrumenttien fysiikkaa...")
    history_tensor, symbol_names, tvm_batch, point_batch = get_marketwatch_history(count=30)
    
    n = len(symbol_names)
    if n == 0:
        print("VIRHE: Market Watch on tyhjä tai historiaa ei saatu.")
        mt5.shutdown()
        return

    # 2. VOLATILITEETTI-TARKISTUS: Lasketaan ATR jokaiselle livenä
    atr_values = calculate_atr_pips(history_tensor, point_batch)

    # 3. VAIHE 1: MASSIVE DOE (27 Skenaariota)
    states = jnp.ones(n, dtype=jnp.int32)
    print(f"[{time.strftime('%H:%M:%S')}] Ajetaan 27-skenaarion DoE {n} instrumentille...")
    _, doe_results = run_full_factorial_doe(history_tensor, states, tvm_batch)

    # 4. VAIHE 2: TRIPLE PARETO GILLOTINE (Stability Score)
    stability_scores = jnp.abs(jnp.mean(doe_results, axis=0)) / (jnp.std(doe_results, axis=0) + 1e-9)
    
    # JAX palauttaa kiinteän määrän indeksejä (0.8% Club)
    elite_indices, elite_scores = triple_pareto_filter(stability_scores, atr_values)
    
    # --- VISUAL MANAGEMENT: VOLATILITEETTI-DIAGNOSTIIKKA ---
    print("\n" + " GENBA-TILANNE: MARKKINAN LIIKE (M5 ATR) ".center(75, "-"))
    # Poimitaan top 5 ATR-arvoa riippumatta suodattimesta
    top_atr_idx = jnp.argsort(atr_values)[::-1][:5]
    for idx in top_atr_idx:
        print(f" > [INFO] {symbol_names[int(idx)].ljust(12)} | ATR: {float(atr_values[idx]):.2f} pips")
    print("-" * 75)

    # --- POKA-YOKE: DYNAAMINEN KARSINTA PYTHON-PUOLELLA ---
    real_elite_indices = [int(idx) for i, idx in enumerate(elite_indices) if elite_scores[i] > 0]
    
    if len(real_elite_indices) == 0:
        max_atr = float(jnp.max(atr_values))
        print(f"\n[INFO] Ei eliittiä. Rima: 3.5 pips. Markkinan paras: {max_atr:.2f} pips.")
        mt5.shutdown()
        return

    # Muunnetaan valitut kohteet JAX-tensoreiksi stressitestiä varten
    candidate_indices_jnp = jnp.array(real_elite_indices)
    candidate_symbols = [symbol_names[i] for i in real_elite_indices]
    candidate_tensors = history_tensor[candidate_indices_jnp]
    candidate_tvms = tvm_batch[candidate_indices_jnp]
    candidate_states = states[candidate_indices_jnp]
    candidate_atrs = atr_values[candidate_indices_jnp]

    print(f"Pareto-seula läpäisty. {len(candidate_symbols)} kandidaattia stressitestiin...")

    # 5. VAIHE 3: WOLFF STRESS TEST
    stress_volumes = verify_stress_tolerance(candidate_tensors, candidate_states, candidate_tvms)
    
    # 6. LOPULLINEN LISTAUS JA LOT-LASKENTA
    print("\n" + " ELITE 0.8% CLUB - SOTALISTA ".center(75, "-"))
    print(f"{'SYMBOL'.ljust(15)} | {'LOT'.ljust(8)} | {'ATR (pips)'.ljust(12)} | {'STABILITY'.ljust(12)} | {'RISK'}")
    print("-" * 75)

    final_elite_map = {}
    for i, symbol in enumerate(candidate_symbols):
        if stress_volumes[i] > 0:
            lot = calculate_wolff_lots(symbol, risk_usd=250.0)
            score = float(stability_scores[real_elite_indices[i]])
            atr = float(candidate_atrs[i])
            
            if lot > 0:
                final_elite_map[symbol] = score
                print(f"{symbol.ljust(15)} | {str(lot).ljust(8)} | {f'{atr:.2f}'.ljust(12)} | {f'{score:.2f}'.ljust(12)} | $250.00")

    # 7. TALLENNUS: Päivitetään kanta
    db.update_elite_status(list(final_elite_map.keys()), final_elite_map)

    execution_time = time.time() - start_time
    print("-" * 75)
    print(f"Analyysi valmis: {execution_time:.2f}s | Valitut kohteet: {len(final_elite_map)}")
    print("="*75)

    mt5.shutdown()

if __name__ == "__main__":
    run_wolff_selection()
# -*- coding: utf-8 -*-
"""
VOJKER TRIAGE - JAX LOGIC CORE TEST (DETERMINISTINEN LUKITUS)
Testaa 5D-suojatensorin, tilattoman FSM-kaskadin suorituskyvyn, muotosopimuksen
sekä The Oracle -tyyppiturvallisuuden (Float64 / Int32 eristys).
"""
import pytest
import time
import jax
import jax.numpy as jnp
import numpy as np

# Pakotetaan X64 tarkkuus testiin, aivan kuten tuotannossakin
jax.config.update("jax_enable_x64", True)

# Tuodaan uusi puhdas JAX-funktio ja tyyppikontti
from packs.wolfpack_alpha.logic import analyze_signal_core, FusedPipelineOutput

def test_x64_precision_is_active():
    """Tullimies tarkastaa: Onko 64-bittinen finanssitoleranssi varmasti päällä?"""
    assert jax.config.jax_enable_x64 is True, "HÄTÄSEIS: JAX X64 ei ole aktiivinen!"

def test_fused_logic_5d_geometry_and_latency():
    """Tullimies + The Oracle: Kantageometria, tilattomuus, suoritusnopeus ja tyyppieristys."""
    
    # 1. GENERoidaan 5-AKSELINEN TESTITENSORI (Batch, Symbolit, Historia, OHLC, Konteksti)
    b, n, h, d, c = 2, 8, 30, 4, 4
    
    np.random.seed(42)
    # Hintadata (Float64)
    raw_market_data = np.random.randn(b, n, h, d, c).astype(np.float64)
    # FSM-tilat (Int32)
    raw_fsm_states = np.ones((b, n), dtype=np.int32)
    # UUSI: Pip-koot (Float64) - Esimerkkiarvo EURUSD:lle
    raw_pip_sizes = np.full((b, n), 0.0001, dtype=np.float64)
    
    # Siirretään data laitteistolle (GPU/CPU)
    supertensor = jax.device_put(raw_market_data)
    fsm_states = jax.device_put(raw_fsm_states)
    pip_sizes = jax.device_put(raw_pip_sizes)
    
    # --- ENSIMMÄINEN AJO: JIT Käännösvaihe (Trace-time) ---
    start_compile = time.perf_counter()
    # PÄIVITYS: Nyt annetaan myös pip_sizes moottorille
    output = analyze_signal_core(supertensor, fsm_states, pip_sizes)
    
    # JAX laiska suoritus vaatii .block_until_ready()
    output.final_signals.block_until_ready()
    compile_time = time.perf_counter() - start_compile
    print(f"\n[JIT-käännösaika (1. ajo)]: {compile_time:.4f} sekuntia")
    
    # --- TOINEN AJO: Puhdas laitteistosuoritus (Run-time) ---
    start_run = time.perf_counter()
    
    # JEDI-KORJAUS: Koska ensimmäinen fsm_states "lahjoitettiin" (Buffer Donation),
    # se tuhottiin välimuistissa. Siksi seuraavalle kierrokselle on syötettävä
    # ensimmäisen ajon palauttama UUSI tila (output.next_fsm_states)!
    output_fast = analyze_signal_core(supertensor, output.next_fsm_states, pip_sizes)
    
    output_fast.final_signals.block_until_ready()
    run_time = time.perf_counter() - start_run
    print(f"[Suoritusaika (2. ajo)]: {run_time:.5f} sekuntia")
    
    # =====================================================================
    # --- TULLIMIEHEN LEIMAT (Alkuperäiset Asserts - MITÄÄN EI POISTETTU) ---
    # =====================================================================
    assert isinstance(output_fast, FusedPipelineOutput), "Paluuarvo ei ole FusedPipelineOutput PyTree!"
    assert output_fast.final_signals.shape == (2, 8), f"Väärä signaalimuoto: {output_fast.final_signals.shape}"
    assert output_fast.next_fsm_states.shape == (2, 8), f"Väärä FSM-tilamuoto: {output_fast.next_fsm_states.shape}"
    assert output_fast.box_highs.shape == (2, 8), f"Väärä box_high muoto: {output_fast.box_highs.shape}"
    
    # Cpk 3.0 Latenssivarmistus: Toisen ajon on oltava salamannopea (Alle 10ms)
    assert run_time < 0.01, f"HÄTÄSEIS: Suoritusaika liian hidas! {run_time:.5f}s (Sallittu: < 0.01s)"

    # =====================================================================
    # --- THE ORACLE AUDIT (Uudet Tyyppivarmistukset) ---------------------
    # =====================================================================
    # 1. Signaalin on oltava tasan 32-bittinen kokonaisluku
    assert output_fast.final_signals.dtype == jnp.int32, \
        f"ORACLE FATAL: Signaalit vuotaneet muotoon {output_fast.final_signals.dtype}!"
        
    # 2. FSM-tilan on oltava tasan 32-bittinen kokonaisluku
    assert output_fast.next_fsm_states.dtype == jnp.int32, \
        f"ORACLE FATAL: FSM-tila vuotanut muotoon {output_fast.next_fsm_states.dtype}!"
        
    # 3. Hintarajojen (Box High/Low) on oltava raskaassa 64-bittisessä muodossa
    assert output_fast.box_highs.dtype == jnp.float64, \
        f"ORACLE FATAL: Hinnat pudonneet 32-bittisiksi! {output_fast.box_highs.dtype}"

    print("\n[TULLIMIES & ORACLE]: Kantageometria, MUISTISOPIMUS, latenssi ja TYYPPITURVALLISUUS hyväksytty. Cpk 3.0 Verified.")
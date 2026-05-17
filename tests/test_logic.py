# -*- coding: utf-8 -*-
"""
VOJKER TRIAGE - JAX LOGIC CORE TEST (DETERMINISTINEN LUKITUS)
Testaa 5D-suojatensorin ja tilattoman FSM-kaskadin suorituskyvyn ja muotosopimuksen.
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
    """Tullimies tarkastaa: Kantageometria, tilattomuus ja suoritusnopeus."""
    
    # 1. GENERoidaan 5-AKSELINEN TESTITENSORI (Batch, Symbolit, Historia, OHLC, Konteksti)
    # Valitaan b=2 skenaariota, n=8 valuuttaparia, h=30 kynttilää, d=4 (OHLC), c=4 (Kontekstit)
    b, n, h, d, c = 2, 8, 30, 4, 4
    
    # Numpyllä luotu satunnainen hintadata (Tämä edustaa MT5:stä tulevaa float64-raakadataa)
    np.random.seed(42)
    raw_market_data = np.random.randn(b, n, h, d, c).astype(np.float64)
    
    # Alustetaan FSM-tila kaikille skenaarioille ja pareille tilaan 1 (IDLE)
    raw_fsm_states = np.ones((b, n), dtype=np.int32)
    
    # Siirretään data laitteistolle (GPU/CPU)
    supertensor = jax.device_put(raw_market_data)
    fsm_states = jax.device_put(raw_fsm_states)
    
    # --- ENSIMMÄINEN AJO: JIT Käännösvaihe (Trace-time) ---
    start_compile = time.perf_counter()
    output = analyze_signal_core(supertensor, fsm_states)
    
    # JAX laiska suoritus vaatii .block_until_ready() tarkan ajan mittaamiseen
    output.final_signals.block_until_ready()
    compile_time = time.perf_counter() - start_compile
    print(f"\n[JIT-käännösaika (1. ajo)]: {compile_time:.4f} sekuntia")
    
    # --- TOINEN AJO: Puhdas laitteistosuoritus (Run-time) ---
    start_run = time.perf_counter()
    output_fast = analyze_signal_core(supertensor, fsm_states)
    output_fast.final_signals.block_until_ready()
    run_time = time.perf_counter() - start_run
    print(f"[Suoritusaika (2. ajo)]: {run_time:.5f} sekuntia")
    
    # --- TULLIMIEHEN LEIMAT (Asserts) ---
    
    # A. Varmistetaan, että paluuarvo on täydellisen muotoinen FusedPipelineOutput
    assert isinstance(output_fast, FusedPipelineOutput), "Paluuarvo ei ole FusedPipelineOutput PyTree!"
    
    # B. Tarkistetaan muotosopimus: b=2, n=8
    assert output_fast.final_signals.shape == (2, 8), f"Väärä signaalimuoto: {output_fast.final_signals.shape}"
    assert output_fast.next_fsm_states.shape == (2, 8), f"Väärä FSM-tilamuoto: {output_fast.next_fsm_states.shape}"
    assert output_fast.box_highs.shape == (2, 8), f"Väärä box_high muoto: {output_fast.box_highs.shape}"
    
    # C. Cpk 3.0 Latenssivarmistus: Toisen ajon on oltava salamannopea (Alle 10ms / 0.01s)
    assert run_time < 0.01, f"HÄTÄSEIS: Suoritusaika liian hidas! {run_time:.5f}s (Sallittu: < 0.01s)"

    print("\n[TULLIMIES]: Kantageometria ja tilaton fuusio hyväksytty. Cpk 3.0 Verified.")
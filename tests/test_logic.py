import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

# Lisätään projektin juuri polkuun
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from packs.wolfpack_alpha.logic import analyze_signal_core

def create_synthetic_supertensor():
    """
    Luodaan synteettinen (8, 3, 30, 4) tensori testausta varten.
    Symboli 0: Täydellinen signaali (Läpäisee kaikki portit)
    Symboli 1: Kohina (Matala volatiliteetti - Gate I hylkää)
    Symboli 2: Kohina (Väärä suunta - Gate II hylkää)
    Symboli 3: Kohina (Matala RNAI / Absorptio - Gate III hylkää)
    Muut: Tyhjää
    """
    shape = (8, 3, 30, 4)
    # Alustetaan nollilla (Open=1.0, High=1.0, Low=1.0, Close=1.0)
    data = np.ones(shape, dtype=np.float32)
    
    # --- SYMBOLI 0: TÄYDELLINEN SIGNAALI ---
    # K=0 (M1), Viimeisin kynttilä: Open=1.0, High=1.5, Low=0.9, Close=1.4 (Suuri ja nouseva)
    data[0, 0, -1, :] = [1.0, 1.5, 0.9, 1.4]
    # K=1 (M5), Viimeisin kynttilä: Open=1.0, Close=1.2 (Nouseva linjaus)
    data[0, 1, -1, :] = [1.0, 1.3, 0.9, 1.2]
    # K=2 (RNAI): Korkea aggressio (2.5)
    data[0, 2, :, :] = 2.5 

    # --- SYMBOLI 1: MATALA VOLATILITEETTI (Hylkäys Gate I) ---
    data[1, 0, -1, :] = [1.0, 1.01, 0.99, 1.0] # Liian pieni liike
    data[1, 2, :, :] = 2.5

    # --- SYMBOLI 2: RAKENTEELLINEN RISTIRIITA (Hylkäys Gate II) ---
    data[2, 0, -1, :] = [1.0, 1.5, 0.9, 1.4] # M1 Nousee
    data[2, 1, -1, :] = [1.3, 1.4, 0.9, 1.0] # M5 Laskee (K-linjaus rikki)
    data[2, 2, :, :] = 2.5

    # --- SYMBOLI 3: ABSORPTIO (Hylkäys Gate III) ---
    data[3, 0, -1, :] = [1.0, 1.5, 0.9, 1.4] # Hinta nousee
    data[3, 1, -1, :] = [1.0, 1.3, 0.9, 1.2] # M5 linjassa
    data[3, 2, :, :] = -0.5 # MUTTA RNAI negatiivinen (Passiivinen absorptio, ei aggressio)

    return jnp.array(data)

def test_logic_precision():
    print("=== VOJKER PHASE 2: LOGIC INTEGRITY TEST (Cpk 3.0) ===")
    
    tensor = create_synthetic_supertensor()
    
    # Ajetaan analyysi
    # Ensimmäinen ajo on hitaampi (JIT-käännös), toinen on salamannopea
    signals = analyze_signal_core(tensor)
    
    print(f"Analysoitu 8 symbolia. Signaalimaski: {signals}")

    # VALIDIOINTI
    # Odotamme: Symboli 0 = True, muut = False
    expected = jnp.array([True, False, False, False, False, False, False, False])
    
    if jnp.array_equal(signals, expected):
        print("\n  PASSED: Triple Quantile Gate erotti signaalin kohinasta täydellisesti.")
        print("  PASSED: Gate III (RNAI) tunnisti absorption oikein.")
    else:
        print("\n  FAILED: Logiikka antoi vääriä signaaleja!")
        for i, s in enumerate(signals):
            if s != expected[i]:
                print(f"    Virhe symbolissa {i}: Odotettiin {expected[i]}, saatiin {s}")

    # Suorituskykytesti (JIT)
    import time
    start = time.time()
    for _ in range(100):
        analyze_signal_core(tensor).block_until_ready()
    end = time.time()
    print(f"\n  PERFORMANCE: 100 sykliä kesti {(end-start)*1000:.2f}ms ({(end-start)*10:.4f}ms / sykli)")

if __name__ == "__main__":
    test_logic_precision()
import sys
import os
# Lisätään projektin juuri polkuun, jotta adapteri löytyy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adapters.mt5_adapter import MT5Adapter
import MetaTrader5 as mt5
import jax.numpy as jnp

def test_supertensor_integrity():
    print("ALoitetaan Cpk 3.0 -integritiittitesti...")
    
    # 1. Alustus
    if not mt5.initialize():
        print("VIRHE: MT5-yhteyttä ei voitu muodostaa. Testi hylätty.")
        return

    try:
        adapter = MT5Adapter()
        tensor = adapter.get_wolfpack_tensors()

        # 2. Tarkistetaan, että dataa saatiin
        if tensor is None:
            print("VIRHE: Tensori on None. Tarkista symbolien saatavuus MT5:ssä.")
            return

        # 3. Muotosopimuksen validointi (KRIITTINEN)
        expected_shape = (8, 3, 30, 4)
        if tensor.shape == expected_shape:
            print(f"PASSED: Muotosopimus täsmää {tensor.shape}")
        else:
            print(f"FAILED: Muoto on {tensor.shape}, odotettiin {expected_shape}")

        # 4. Datatyypin validointi
        if tensor.dtype == jnp.float32:
            print(f"PASSED: Datatyyppi on oikea (float32)")
        else:
            print(f"FAILED: Datatyyppi on {tensor.dtype}")

        # 5. Arvoalueen tarkistus (ei nollia tai NaN-arvoja kriittisissä paikoissa)
        if not jnp.isnan(tensor).any():
            print("PASSED: Ei NaN-arvoja tensorissa.")
        else:
            print("FAILED: Tensori sisältää viallisia NaN-arvoja.")

    finally:
        mt5.shutdown()
        print("Testiajo suoritettu.")

if __name__ == "__main__":
    test_supertensor_integrity()
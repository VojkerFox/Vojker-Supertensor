# -*- coding: utf-8 -*-
import sys
import os
import unittest
import numpy as np
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from packs.wolfpack_alpha.logic import analyze_signal_core

class Test8PairSymmetry(unittest.TestCase):
    
    def create_perfect_tensor(self):
        # Luodaan standardi (8, 3, 30, 4) tensori
        tensor = np.zeros((8, 3, 30, 4), dtype=np.float32)
        
        # Asetetaan kaikille perusvolatiliteetti (estetään Gate 1 hylkäys)
        # Keskiarvo 0.00010, nykyinen 0.00020 (eli > 1.5x)
        tensor[:, 0, :, 1] = 0.00010 # Highs
        tensor[:, 0, :, 2] = 0.00000 # Lows
        tensor[:, 0, -1, 1] = 0.00020 # Viimeisin High
        
        return tensor

    def test_8_pair_logic_flow(self):
        raw_tensor = self.create_perfect_tensor()
        
        # Rakennetaan testiskenaariot kaikille 8 indeksille:
        # Indeksit 0, 2, 4, 6: LONG (Gate 2: C>O, Gate 3: RNAI > 1.0)
        # Indeksit 1, 3, 5, 7: SHORT (Gate 2: C<O, Gate 3: RNAI < -1.0)
        
        for i in range(8):
            if i % 2 == 0: # LONG
                raw_tensor[i, 0, -1, 3] = 1.1005 # Close
                raw_tensor[i, 0, -1, 0] = 1.1000 # Open
                raw_tensor[i, 1, -1, 3] = 1.1005 # M5 Close
                raw_tensor[i, 1, -1, 0] = 1.1000 # M5 Open
                raw_tensor[i, 2, -1, 0] = 1.5    # RNAI
            else: # SHORT
                raw_tensor[i, 0, -1, 3] = 1.0995 # Close
                raw_tensor[i, 0, -1, 0] = 1.1000 # Open
                raw_tensor[i, 1, -1, 3] = 1.0995 # M5 Close
                raw_tensor[i, 1, -1, 0] = 1.1000 # M5 Open
                raw_tensor[i, 2, -1, 0] = -1.5   # RNAI

        jnp_tensor = jnp.array(raw_tensor)
        signals, box_highs, box_lows = analyze_signal_core(jnp_tensor)

        print(f"\n--- 8-PARIN SYMMETRIATESTI ---")
        print(f"Signaalimaski: {signals}")

        # Varmistetaan että jokainen kanava antoi signaalin
        for i in range(8):
            expected = 1 if i % 2 == 0 else 2
            self.assertEqual(signals[i], expected, f"Symboli {i} epäonnistui. Odotettiin {expected}, saatiun {signals[i]}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
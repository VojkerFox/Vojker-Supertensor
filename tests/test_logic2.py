# -*- coding: utf-8 -*-
import sys
import os
import unittest
import numpy as np
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from packs.wolfpack_alpha.logic import analyze_signal_core

class TestVojkerPhysics(unittest.TestCase):
    
    def create_synthetic_tensor(self, m5_res, m5_sup, m1_high, m1_low, rnai_val):
        tensor = np.zeros((8, 3, 30, 4), dtype=np.float32)
        
        # Asetetaan M5 vastus ja tuki
        tensor[0, 1, :-1, 1] = m5_res
        tensor[0, 1, :-1, 2] = m5_sup
        
        # Asetetaan M1 nykyinen kynttilä
        tensor[0, 0, -1, 1] = m1_high
        tensor[0, 0, -1, 2] = m1_low
        
        # Asetetaan M1 historia Salaman laatikkoa varten
        tensor[0, 0, -5:-1, 1] = m5_res - 0.0001
        tensor[0, 0, -5:-1, 2] = m5_sup + 0.0001
        
        # Asetetaan RNAI-aggressio
        tensor[0, 2, -1, 0] = rnai_val
        
        return jnp.array(tensor)

    def test_01_perfect_long_breakout(self):
        m5_res = 1.10000
        # EXACT TOUCH: m1_low on tasan m5_res, jolloin float32-pyöristys ei voi pilata ehtoa
        tensor = self.create_synthetic_tensor(
            m5_res=m5_res, m5_sup=1.09000, 
            m1_high=1.10050,  # Break
            m1_low=1.10000,   # Exact Retest Touch
            rnai_val=1.5      # Hyvä RNAI
        )
        
        # DEBUG-OSIO: Tulostetaan JAX-matematiikka livenä
        m1 = tensor[0, 0]
        m5 = tensor[0, 1]
        rnai = tensor[0, 2, -1, 0]
        calc_m5_res = jnp.max(m5[:-1, 1])
        break_long = m1[-1, 1] > calc_m5_res
        retest_long = m1[-1, 2] <= (calc_m5_res + 0.00002)
        
        print("\n--- DEBUG LONG ---")
        print(f"M5 Res: {calc_m5_res}")
        print(f"Break Long (High > Res): {m1[-1, 1]} > {calc_m5_res} -> {break_long}")
        print(f"Retest Long (Low <= Res+Buffer): {m1[-1, 2]} <= {calc_m5_res + 0.00002} -> {retest_long}")
        print(f"RNAI (> 0.8): {rnai} -> {rnai > 0.8}")
        
        signals, box_highs, box_lows = analyze_signal_core(tensor)
        self.assertEqual(signals[0], 1, "Täydellinen LONG pitäisi laukaista signaali 1.")

    def test_02_perfect_short_breakout(self):
        m5_sup = 1.09000
        tensor = self.create_synthetic_tensor(
            m5_res=1.10000, m5_sup=m5_sup, 
            m1_high=1.09000,  # Exact Retest Touch
            m1_low=1.08950,   # Break
            rnai_val=-1.5     
        )
        signals, box_highs, box_lows = analyze_signal_core(tensor)
        self.assertEqual(signals[0], 2, "Täydellinen SHORT pitäisi laukaista signaali 2.")

    def test_03_failed_retest_no_signal(self):
        m5_res = 1.10000
        tensor = self.create_synthetic_tensor(
            m5_res=m5_res, m5_sup=1.09000, 
            m1_high=1.10050,  
            m1_low=1.10005,   # Fails Retest
            rnai_val=1.5      
        )
        signals, box_highs, box_lows = analyze_signal_core(tensor)
        self.assertEqual(signals[0], 0, "Failed Retest pitäisi pitää signaali nollassa.")

    def test_04_low_rnai_no_signal(self):
        m5_res = 1.10000
        tensor = self.create_synthetic_tensor(
            m5_res=m5_res, m5_sup=1.09000, 
            m1_high=1.10050,  
            m1_low=1.10000,   
            rnai_val=0.5      # Fails RNAI
        )
        signals, box_highs, box_lows = analyze_signal_core(tensor)
        self.assertEqual(signals[0], 0, "Heikko RNAI pitäisi hylätä kauppa.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
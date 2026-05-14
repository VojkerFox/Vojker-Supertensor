# -*- coding: utf-8 -*-
import sys
import os
import unittest
import numpy as np
import jax.numpy as jnp

# Tämä rivi varmistaa, että löydämme alkuperäisen koodin rikkomatta sitä
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from packs.wolfpack_alpha.logic import analyze_signal_core

class TestVojkerPhysics(unittest.TestCase):
    
    def test_01_perfect_long_breakout(self):
        # 1. Luodaan tyhjä 4D tensori (Synteettinen Data)
        # Muoto: (1 symboli, 3 kanavaa, 30 kynttilää, 4 datapunktia)
        # Käytetään 1 symbolia 8 sijaan testin yksinkertaistamiseksi.
        # Muutetaan analyze_signal_core käsittelemään yksittäistä symbolia 
        # testin takia (normaalisti se vmap:taa 8 yli)
        
        # Oletetaan, että M5_res on 1.1000 ja M5_sup on 1.0900
        # Oletetaan, että nykyinen M1 on murtanut ja retestannut
        tensor = np.zeros((1, 3, 30, 4), dtype=np.float32)
        
        # M5 High (Vastus)
        tensor[0, 1, :-1, 1] = 1.1000
        # M5 Low (Tuki)
        tensor[0, 1, :-1, 2] = 1.0900
        
        # M1 Nykyinen kynttilä
        tensor[0, 0, -1, 1] = 1.1005 # Break (High > M5_res)
        tensor[0, 0, -1, 2] = 1.1000 # Exact Retest (Low = M5_res)
        
        # RNAI
        tensor[0, 2, -1, 0] = 1.5    # Vahva voima (> 0.8)
        
        jnp_tensor = jnp.array(tensor)
        
        # Kutsumme prosessointia suoraan yhdelle symbolille, koska synteettinen 
        # tensori on vain muotoa (1, ...). 
        # HUOM: Testissä ei kutsuta jax.vmap:ttua ydinfunktiota, vaan yksittäistä logiikkaa
        from packs.wolfpack_alpha.logic import process_symbol_logic
        signals, box_highs, box_lows = process_symbol_logic(jnp_tensor[0])
        
        # 3. Varmistus
        self.assertEqual(signals, 1, "Täydellinen LONG ei laukaissut signaalia.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
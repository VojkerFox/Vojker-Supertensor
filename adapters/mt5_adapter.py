# -*- coding: utf-8 -*-
"""
VOJKER ADAPTER: MetaTrader 5 & Quantum Ingest
Purpose: Construct the 4D Supertensor (8, 3, 30, 4)
"""
import MetaTrader5 as mt5
import jax.numpy as jnp
import numpy as np

# Määritellään Wolfpack-symbolit (8 kpl)
WOLFPACK_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "EURJPY"]

class MT5Adapter:
    def __init__(self, symbols=WOLFPACK_SYMBOLS, bars=30):
        self.symbols = symbols
        self.bars = bars

    def get_wolfpack_tensors(self, quantum_context=None):
        """
        Rakentaa 4D-supertensorin.
        K=0: M1 (Aika)
        K=1: M5 (Rakenne)
        K=2: Quantum (IQM Sirius / Relevance Axis)
        """
        all_symbol_data = []
        
        for s in self.symbols:
            # Haetaan M1 ja M5 kynttilädata
            r_m1 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M1, 0, self.bars)
            r_m5 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, self.bars)
            
            if r_m1 is None or r_m5 is None or len(r_m1) < self.bars:
                return None
            
            # Poimitaan OHLC (Open, High, Low, Close)
            m1_data = np.array([[r[1], r[2], r[3], r[4]] for r in r_m1], dtype=np.float32)
            m5_data = np.array([[r[1], r[2], r[3], r[4]] for r in r_m5], dtype=np.float32)
            
            # K=2 Kvantti-akselin rakentaminen
            # Jos API:sta saadaan arvo (esim. 0.984), täytetään se koko akselille
            q_val = quantum_context.get(s, 1.0) if quantum_context else 1.0
            q_data = np.ones_like(m1_data, dtype=np.float32) * q_val
            
            # Pinotaan K-akselit (3, 30, 4)
            symbol_stack = np.stack([m1_data, m5_data, q_data], axis=0)
            all_symbol_data.append(symbol_stack)
            
        # Palautetaan lopullinen Supertensori (8, 3, 30, 4) JAX-muodossa
        return jnp.array(all_symbol_data) if len(all_symbol_data) == 8 else None

if __name__ == "__main__":
    # Testiajo
    if mt5.initialize():
        adapter = MT5Adapter()
        tensor = adapter.get_wolfpack_tensors()
        if tensor is not None:
            print(f"SUUNNITELMA: Supertensori muodostettu. Shape: {tensor.shape}")
            # Varmistetaan Raw Data -todiste
            print(f"RAW DATA PROOF (EURUSD M1 last close): {tensor[0, 0, -1, 3]}")
        mt5.shutdown()
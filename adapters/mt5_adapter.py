# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import jax.numpy as jnp
import numpy as np

WOLFPACK_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "USDCHF", "EURJPY"]

class MT5Adapter:
    def __init__(self, symbols=WOLFPACK_SYMBOLS, bars=30):
        self.symbols = symbols
        self.bars = bars

    def get_wolfpack_tensors(self, quantum_context=None):
        raw_symbol_containers = []
        symbol_aggressions = []
        
        # VAIHE 1: Kerätään raakadata ja lasketaan symbolikohtainen aggressio
        for s in self.symbols:
            r_m1 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M1, 0, self.bars)
            r_m5 = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, self.bars)
            
            if r_m1 is None or r_m5 is None or len(r_m1) < self.bars:
                return None
            
            # OHLC Data
            m1_data = np.array([[r[1], r[2], r[3], r[4]] for r in r_m1], dtype=np.float32)
            m5_data = np.array([[r[1], r[2], r[3], r[4]] for r in r_m5], dtype=np.float32)
            
            # RNAI PROXY: (Close - Open) * TickVolume
            # Tämä mittaa "aggressiivista vaivaa" suhteessa lopputulokseen [cite: 44, 61, 107]
            last_candle = r_m1[-1]
            net_aggression = (last_candle[4] - last_candle[1]) * last_candle[5] # 5 = tick_volume
            symbol_aggressions.append(net_aggression)
            
            # Tallennetaan välivarastoon
            raw_symbol_containers.append((m1_data, m5_data))

        # VAIHE 2: Markkinan keskiarvon laskenta (The Market Context)
        market_avg_aggression = np.mean(symbol_aggressions)
        
        # VAIHE 3: Rakennetaan lopullinen 4D-supertensori
        all_symbol_data = []
        for i, (m1_data, m5_data) in enumerate(raw_symbol_containers):
            # Lasketaan suhteellinen aggressio (RNAI)
            # Jos symboli on aggressiivisempi kuin markkina keskimäärin, se on Alpha-signaali [cite: 107]
            rnai_val = symbol_aggressions[i] - market_avg_aggression
            
            # K=2 täytetään RNAI-arvolla
            q_data = np.ones_like(m1_data, dtype=np.float32) * rnai_val
            
            # Pinotaan sopimuksen mukaisesti (3, 30, 4)
            symbol_stack = np.stack([m1_data, m5_data, q_data], axis=0)
            all_symbol_data.append(symbol_stack)
            
        return jnp.array(all_symbol_data) if len(all_symbol_data) == 8 else None
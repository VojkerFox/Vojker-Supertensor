# -*- coding: utf-8 -*-
"""
VOJKER RDAAS FSM: The Panama Canal 6-Step Cycle (Cpk 3.0)
Purpose: Deterministic state management for the Wolfpack (8 symbols).
Following the "Surgical Intervention" protocol with asymmetric risk/reward.
"""
import jax.numpy as jnp

class PanamaFSM:
    # Infograafin mukaiset 6 tilaa
    NEUTRAL = 0   # Järjestelmän nollaus / Cooldown
    IDLE = 1      # Aktiivinen haku (Scanning Market)
    ARMED = 2     # Signal Core löydetty (Outlier Discovery)
    ACTION = 3    # Triggeröinti (Execution)
    MANAGE = 4    # Position hallinta (Risk Management)
    EXIT = 5      # Sulkemisprosessi (Cleanup)

    def __init__(self):
        # Alustetaan tilat ja lukitushinnat kaikille 8 symbolille
        self.states = jnp.zeros(8, dtype=jnp.int32)
        self.lock_prices = jnp.zeros(8, dtype=jnp.float32)
        self.directions = jnp.zeros(8, dtype=jnp.int32) # 1=LONG, -1=SHORT, 0=NONE
        self.sl_prices = jnp.zeros(8, dtype=jnp.float32) # Stop Loss -tasot
        self.box_sizes = jnp.zeros(8, dtype=jnp.float32) # Salaman laatikoiden koot

    def update(self, signal_mask, box_highs, box_lows, current_prices, current_rnai_values):
        """
        Päivittää jokaisen symbolin tilan perustuen:
        - logic.py:n antamaan signaaliin ja Salaman laatikon rajoihin
        - Nykyiseen hintaan (Salaman murtuma)
        - RNAI-arvoon (Aggressio-varmistus ja trailing)
        """
        actions = []

        for i in range(8):
            state = self.states[i]
            signal = signal_mask[i]
            price = current_prices[i]
            rnai = current_rnai_values[i]
            direction = self.directions[i]
            
            # Puskuri: 0.35 pipiä (5 desimaalin hinnoittelussa 0.000035)
            BUFFER = 0.000035

            # 1. NEUTRAL -> IDLE: Varmistetaan puhdas aloitus
            if state == self.NEUTRAL:
                self.states = self.states.at[i].set(self.IDLE)
                self.directions = self.directions.at[i].set(0)
                actions.append(f"FSM_{i}: Initialized to IDLE")

            # 2. IDLE -> ARMED: Salaman laatikko tunnistettu ja lukittu
            elif state == self.IDLE:
                if signal == 1: # LONG
                    self.states = self.states.at[i].set(self.ARMED)
                    self.lock_prices = self.lock_prices.at[i].set(box_highs[i]) # Entry katolta
                    self.sl_prices = self.sl_prices.at[i].set(box_lows[i] - BUFFER) # SL laatikon alle
                    self.box_sizes = self.box_sizes.at[i].set(box_highs[i] - box_lows[i] + BUFFER)
                    self.directions = self.directions.at[i].set(1)
                    actions.append(f"FSM_{i}: LONG_LOCKED at {box_highs[i]:.5f}. SL: {box_lows[i]-BUFFER:.5f}")
                elif signal == 2: # SHORT
                    self.states = self.states.at[i].set(self.ARMED)
                    self.lock_prices = self.lock_prices.at[i].set(box_lows[i]) # Entry lattiasta
                    self.sl_prices = self.sl_prices.at[i].set(box_highs[i] + BUFFER) # SL laatikon päälle
                    self.box_sizes = self.box_sizes.at[i].set(box_highs[i] - box_lows[i] + BUFFER)
                    self.directions = self.directions.at[i].set(-1)
                    actions.append(f"FSM_{i}: SHORT_LOCKED at {box_lows[i]:.5f}. SL: {box_highs[i]+BUFFER:.5f}")
                else:
                    actions.append(None)

            # 3. ARMED -> ACTION: The Trigger (1.0 pip break WITH aggression)
            elif state == self.ARMED:
                if direction == 1: # LONG
                    price_break = price > self.lock_prices[i]
                    exhaustion = rnai < 0.2
                    aggression_ok = rnai > 1.0
                else: # SHORT
                    price_break = price < self.lock_prices[i]
                    exhaustion = rnai > -0.2
                    aggression_ok = rnai < -1.0
                
                # RNAI-varmistus: Jos aggressio kuolee, kyseessä on Exhaustion
                if exhaustion: 
                    self.states = self.states.at[i].set(self.NEUTRAL)
                    actions.append(f"FSM_{i}: CANCEL - Aggression decayed (Exhaustion).")
                # Jos hinta murtuu ja aggressio on yhä korkea
                elif price_break and aggression_ok:
                    self.states = self.states.at[i].set(self.ACTION)
                    actions.append(f"FSM_{i}: EXECUTE - Box broken with high RNAI.")
                # Jos alkuperäinen signaali katoaa, palataan IDLE-tilaan
                elif not signal:
                    self.states = self.states.at[i].set(self.IDLE)
                    actions.append(f"FSM_{i}: ARMED signal lost. Resetting to IDLE.")
                else:
                    actions.append(None)

            # 4. ACTION -> MANAGE: Siirrytään hallintavaiheeseen
            elif state == self.ACTION:
                self.states = self.states.at[i].set(self.MANAGE)
                actions.append(f"FSM_{i}: Position Active. Moving to MANAGE.")

            # 5. MANAGE -> EXIT: Seurataan poistumisehtoja (2R BE+0.35, SL ja Trailing RNAI)
            elif state == self.MANAGE:
                box_size = self.box_sizes[i]
                
                if direction == 1: # LONG
                    profit = price - self.lock_prices[i]
                    
                    # 2R Saavutettu -> Siirretään Stop Loss Break Eveniin + 0.35 pipiä
                    if profit >= (2 * box_size) and self.sl_prices[i] < self.lock_prices[i]:
                        self.sl_prices = self.sl_prices.at[i].set(self.lock_prices[i] + BUFFER)
                        actions.append(f"FSM_{i}: 2R REACHED! SL moved to BE+0.35")
                        continue
                        
                    sl_hit = price < self.sl_prices[i]
                    absorption_exit = rnai < -1.0
                    
                else: # SHORT
                    profit = self.lock_prices[i] - price
                    
                    # 2R Saavutettu -> Siirretään Stop Loss Break Eveniin + 0.35 pipiä
                    if profit >= (2 * box_size) and self.sl_prices[i] > self.lock_prices[i]:
                        self.sl_prices = self.sl_prices.at[i].set(self.lock_prices[i] - BUFFER)
                        actions.append(f"FSM_{i}: 2R REACHED! SL moved to BE+0.35")
                        continue
                        
                    sl_hit = price > self.sl_prices[i]
                    absorption_exit = rnai > 1.0
                    
                if sl_hit or absorption_exit:
                    reason = "SL Hit" if sl_hit else "Absorption Exit"
                    self.states = self.states.at[i].set(self.EXIT)
                    actions.append(f"FSM_{i}: Exit condition met ({reason}). Moving to EXIT.")
                else:
                    actions.append(None)

            # 6. EXIT -> NEUTRAL: Syklin puhdistus
            elif state == self.EXIT:
                self.states = self.states.at[i].set(self.NEUTRAL)
                actions.append(f"FSM_{i}: Cleanup complete. Returning to NEUTRAL.")

        return actions
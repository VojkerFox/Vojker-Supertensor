# -*- coding: utf-8 -*-
"""
VOJKER RDAAS FSM: The Panama Canal 6-Step Cycle
Purpose: Deterministic state management for the Wolfpack (8 symbols).
Following the "Surgical Intervention" protocol.
"""
import jax.numpy as jnp

class PanamaFSM:
    # Infograafin mukaiset 6 tilaa
    NEUTRAL = 0   # Järjestelmän nollaus / Cooldown
    IDLE = 1      # Aktiivinen haku (Scanning Market)
    ARMED = 2     # Signal Core löydetty (Outlier Discovery)
    ACTION = 3    # Trigeröinti (Execution)
    MANAGE = 4    # Position hallinta (Risk Management)
    EXIT = 5      # Sulkemisprosessi (Cleanup)

    def __init__(self):
        # Alustetaan tilat ja lukitushinnat kaikille 8 symbolille
        self.states = jnp.zeros(8, dtype=jnp.int32)
        self.lock_prices = jnp.zeros(8, dtype=jnp.float32)
        self.directions = jnp.zeros(8, dtype=jnp.int32) # Lisätty: 1=LONG, -1=SHORT, 0=NONE

    def update(self, signal_mask, current_prices, current_rnai_values):
        """
        Päivittää jokaisen symbolin tilan perustuen:
        - logic.py:n antamaan signaaliin (Signal Core)
        - Nykyiseen hintaan (1.0 pip entry)
        - RNAI-arvoon (Aggressio-varmistus)
        """
        actions = []

        for i in range(8):
            state = self.states[i]
            signal = signal_mask[i]
            price = current_prices[i]
            rnai = current_rnai_values[i]
            direction = self.directions[i]

            # 1. NEUTRAL -> IDLE: Varmistetaan puhdas aloitus
            if state == self.NEUTRAL:
                self.states = self.states.at[i].set(self.IDLE)
                self.directions = self.directions.at[i].set(0)
                actions.append(f"FSM_{i}: Initialized to IDLE")

            # 2. IDLE -> ARMED: Outlier Discovery (Logic.py löysi 0.8% signaalin)
            elif state == self.IDLE:
                if signal == 1: # LONG
                    self.states = self.states.at[i].set(self.ARMED)
                    self.lock_prices = self.lock_prices.at[i].set(price)
                    self.directions = self.directions.at[i].set(1)
                    actions.append(f"FSM_{i}: LONG_LOCKED at {price}. Moving to ARMED.")
                elif signal == 2: # SHORT
                    self.states = self.states.at[i].set(self.ARMED)
                    self.lock_prices = self.lock_prices.at[i].set(price)
                    self.directions = self.directions.at[i].set(-1)
                    actions.append(f"FSM_{i}: SHORT_LOCKED at {price}. Moving to ARMED.")
                else:
                    actions.append(None)

            # 3. ARMED -> ACTION: The Trigger (1.0 pip break WITH aggression)
            elif state == self.ARMED:
                # Odotetaan dynaamista 1.0 pipin murtumaa SUUNNAN MUKAAN
                if direction == 1: # LONG
                    price_break = price > (self.lock_prices[i] + 0.0001)
                    exhaustion = rnai < 0.2
                    aggression_ok = rnai > 1.0
                else: # SHORT
                    price_break = price < (self.lock_prices[i] - 0.0001)
                    exhaustion = rnai > -0.2
                    aggression_ok = rnai < -1.0
                
                # RNAI-varmistus: Jos aggressio kuolee, kyseessä on Exhaustion
                if exhaustion: 
                    self.states = self.states.at[i].set(self.NEUTRAL)
                    actions.append(f"FSM_{i}: CANCEL - Aggression decayed (Exhaustion).")
                # Jos hinta murtuu ja aggressio (Effort) on yhä korkea
                elif price_break and aggression_ok:
                    self.states = self.states.at[i].set(self.ACTION)
                    actions.append(f"FSM_{i}: EXECUTE - 1.0 pip break with high RNAI.")
                # Jos alkuperäinen signaali katoaa, palataan IDLE-tilaan
                elif not signal:
                    self.states = self.states.at[i].set(self.IDLE)
                    actions.append(f"FSM_{i}: ARMED signal lost. Resetting to IDLE.")
                else:
                    actions.append(None)

            # 4. ACTION -> MANAGE: Siirrytään hallintavaiheeseen
            elif state == self.ACTION:
                # Tässä vaiheessa MT5-adapteri suorittaa varsinaisen oston/myynnin
                self.states = self.states.at[i].set(self.MANAGE)
                actions.append(f"FSM_{i}: Position Active. Moving to MANAGE.")

            # 5. MANAGE -> EXIT: Seurataan poistumisehtoja (Target/Stop)
            elif state == self.MANAGE:
                # Esimerkki: Exit jos hinta palaa lock-tasolle (Break-even)
                # Tai jos RNAI kääntyy rajusti vastasuuntaan (Absorption)
                if direction == 1: # LONG
                    exit_trigger = (price < self.lock_prices[i]) or (rnai < -1.0)
                else: # SHORT
                    exit_trigger = (price > self.lock_prices[i]) or (rnai > 1.0)
                    
                if exit_trigger:
                    self.states = self.states.at[i].set(self.EXIT)
                    actions.append(f"FSM_{i}: Exit condition met. Moving to EXIT.")
                else:
                    actions.append(None)

            # 6. EXIT -> NEUTRAL: Syklin puhdistus
            elif state == self.EXIT:
                self.states = self.states.at[i].set(self.NEUTRAL)
                actions.append(f"FSM_{i}: Cleanup complete. Returning to NEUTRAL.")

        return actions
# -*- coding: utf-8 -*-
import time
import MetaTrader5 as mt5
import jax.numpy as jnp
from datetime import datetime
from packs.the_accountant.sniffer import calculate_wolff_lots
from packs.the_accountant.executor import open_test_trade

# --- WOLFF QUALITY STANDARDS ---
RISK_PER_TRADE = 250.0

active_positions = {}

def get_pip_unit(info):
    """Dynaaminen pip-koko JPY:lle (3/2 desimaalia) ja muille."""
    return 0.01 if info.digits <= 3 else 0.0001

def forward_test_loop():
    print("\n" + "="*85)
    print(" THE ACCOUNTANT: PURE PRICE ACTION (M5BOS + M1 RETEST + 1:2 RR) ".center(85, "="))
    print("="*85)

    if not mt5.initialize(): return

    acc = mt5.account_info()
    if acc:
        print(f"TILI: {acc.login} | SALDO: {acc.balance} {acc.currency} | VIPU: 1:{acc.leverage}")

    # Otetaan vain ne parit jotka on näkyvissä Market Watchissa
    symbols = [s.name for s in mt5.symbols_get() if s.select]
    print(f"Genba-valmius: {len(symbols)} symbolia seurannassa.")

    while True:
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            for symbol in symbols:
                # 1. DATAN HAKU
                rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 30)
                rates_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 30)
                if rates_m5 is None or rates_m1 is None: continue
                
                info = mt5.symbol_info(symbol)
                tick = mt5.symbol_info_tick(symbol)
                if not tick or not info: continue

                # Rakennetaan tensorit PA-analyysiin
                m5 = jnp.array([[r['high'], r['low'], r['close']] for r in rates_m5])
                m1 = jnp.array([[r['high'], r['low'], r['close']] for r in rates_m1])
                curr_price = tick.ask if m1[-1, 2] > m1[-2, 2] else tick.bid
                pip_val = get_pip_unit(info)

                if symbol not in active_positions:
                    # --- VAIHE 1: M5 BOS ---
                    m5_high, m5_low = jnp.max(m5[:-1, 0]), jnp.min(m5[:-1, 1])
                    m5_bos_up = m5[-1, 2] > m5_high
                    m5_bos_down = m5[-1, 2] < m5_low

                    # --- VAIHE 2: M1 RAKENNE & TRIGGER (+1 PIP) ---
                    m1_h, m1_l = jnp.max(m1[-6:-1, 0]), jnp.min(m1[-6:-1, 1])
                    
                    is_long = jnp.logical_and(m5_bos_up, curr_price > (m1_h + pip_val))
                    is_short = jnp.logical_and(m5_bos_down, curr_price < (m1_l - pip_val))

                    if is_long or is_short:
                        direction = "BUY" if is_long else "SELL"
                        
                        # --- VAIHE 3: SL (M1-Rakenne) & TP (1:2) ---
                        sl = float(m1_l if is_long else m1_h)
                        
                        # Broker Safety: Varmistetaan minimietäisyys (StopLevel)
                        min_stop = info.stops_level * info.point
                        risk_dist = abs(curr_price - sl)
                        
                        if risk_dist < min_stop:
                            sl = curr_price - (min_stop + pip_val) if is_long else curr_price + (min_stop + pip_val)
                            risk_dist = abs(curr_price - sl)

                        tp = float(curr_price + (risk_dist * 2) if is_long else curr_price - (risk_dist * 2))
                        lot = calculate_wolff_lots(symbol, RISK_PER_TRADE)

                        # --- ENTRY ---
                        ticket = open_test_trade(symbol, direction, lot, sl=sl, tp=tp)
                        if ticket:
                            print(f"\n[{current_time}] >>> ENTRY: {symbol} | SL: {sl:.5f} | TP: {tp:.5f} | RR: 1:2")
                            active_positions[symbol] = {"ticket": ticket, "sl": sl, "tp": tp, "type": direction}

                elif symbol in active_positions:
                    # Virtuaalinen siivous Accountantin muistista
                    pos = active_positions[symbol]
                    if (pos['type'] == "BUY" and (curr_price <= pos['sl'] or curr_price >= pos['tp'])) or \
                       (pos['type'] == "SELL" and (curr_price >= pos['sl'] or curr_price <= pos['tp'])):
                        del active_positions[symbol]

            print(f"[{current_time}] Skannataan {len(symbols)} symbolia. Aktiiviset: {len(active_positions)}", end='\r')
            time.sleep(1)

        except Exception as e:
            print(f"\nAUDIT VIRHE: {e}")
            time.sleep(2)

    mt5.shutdown()

if __name__ == "__main__":
    forward_test_loop()
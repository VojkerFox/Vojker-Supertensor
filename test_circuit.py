# test_circuit.py
from packs.the_accountant.executor import open_test_trade, close_test_trade
import MetaTrader5 as mt5
import time

TARGET_SYMBOL = "EURUSD" # Varmista että tämä on Market Watchissa!

print("=== KÄYNNISTETÄÄN PUTKISTOTESTI (AVAUS -> VIIVE -> SULKU) ===")

# 1. AVAUS
ticket = open_test_trade(TARGET_SYMBOL, "BUY", lot=0.01)

if ticket:
    print(f"Odotetaan 5 sekuntia... Katso MT5-terminaalia!")
    time.sleep(5)
    
    # 2. SULKU
    close_test_trade(ticket)

print("=== TESTI VALMIS ===")
mt5.shutdown()
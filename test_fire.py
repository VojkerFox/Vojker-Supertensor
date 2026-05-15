# test_fire.py
from packs.the_accountant.executor import open_test_trade
import MetaTrader5 as mt5

# VALITSE KOHDE SOTALISTALTA
target = "EURUSD" # Tai US30, JP225 jne.

print(f"KÄYNNISTETÄÄN GENBA-TESTI: {target}")
open_test_trade(target, "BUY", lot=0.01)

mt5.shutdown()